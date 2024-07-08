#!/usr/bin/env python

# Copyright 2024 Kemal Kurniawan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from flair.datasets.sequence_labeling import JsonlCorpus
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from tqdm import tqdm

from run_span_predictor import read_jsonl

ex = Experiment("span-prediction")
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore[assignment]
if "SACRED_MONGO_URL" in os.environ:
    ex.observers.append(
        MongoObserver(
            url=os.environ["SACRED_MONGO_URL"],
            db_name=os.getenv("SACRED_DB_NAME", "sacred"),
        )
    )


@ex.config
def default():
    # prediction file in CoNLL format
    pred_path = "test.tsv"
    # whether evaluation is raw (B- and I- tags are stripped)
    is_raw = False
    # whether to return evaluation result (True for testing)
    return_result = False
    # perform ROUGE-like multi-reference evaluation with this disaggregated test set in JSONL format
    disagg_test_path = ""


def _parse_col(col: str) -> float:
    s = col.split(":")[1].lstrip()
    if s.endswith("%"):
        s = s[:-1]
    return float(s)


@ex.capture
def _run_eval(pred_path, is_raw=False, _log=None):
    sep_token = "<sep>"
    with open(pred_path, encoding="utf8") as f:
        lines = f.readlines()
    text_lines = []
    i = 0
    while i < len(lines):
        tokens = []
        j = i
        while (
            sep_token.startswith("".join(tokens))
            and "".join(tokens) != sep_token
            and j < len(lines)
            and (cols := lines[j].strip().split())
        ):
            tokens.append(cols[0])
            j += 1
        if "".join(tokens) != sep_token:
            text_lines.append(lines[i])
            i += 1
        else:
            while i < len(lines) and lines[i].strip():
                i += 1
            if i < len(lines):
                text_lines.append(lines[i])
                i += 1

    args = ["./conlleval"]

    if is_raw:
        args.append("-r")
        clean_text_lines = []
        for line in text_lines:
            cols = line.strip().split()
            if cols:
                if cols[1][:2] in ("B-", "I-"):
                    true = cols[1][2:]
                else:
                    true = cols[1]
                if cols[2][:2] in ("B-", "I-"):
                    pred = cols[2][2:]
                else:
                    pred = cols[2]
                clean_line = " ".join([cols[0], true, pred])
            else:
                clean_line = ""
            clean_text_lines.append(f"{clean_line}\n")
        text_lines = clean_text_lines

    res = subprocess.run(
        args,
        input="".join(text_lines),
        text=True,
        capture_output=True,
        encoding="utf8",
        check=True,
    )
    if _log is not None:
        _log.info("Evaluation result:\n%s", res.stdout)
    cols = res.stdout.splitlines()[1].rstrip().split(";")
    return [_parse_col(c) / 100 for c in cols]


@ex.automain
def evaluate(
    pred_path,
    is_raw=False,
    disagg_test_path=None,
    return_result=True,
    _log=None,
    _run=None,
):
    """Evaluate test span predictions."""
    sep_token = "<sep>"
    if not disagg_test_path:
        values = _run_eval(pred_path, is_raw)
    else:
        test_data = list(read_jsonl(Path(disagg_test_path), sep_token))
        with tempfile.TemporaryDirectory() as tmpdirname:
            with (Path(tmpdirname) / "test.jsonl").open("w", encoding="utf8") as f:
                for dat in test_data:
                    print(json.dumps(dat), file=f)
            corpus = JsonlCorpus(
                tmpdirname,
                text_column_name="text",
                label_column_name="spans",
                label_type="evidence?",
                sample_missing_splits=False,
            )
        tokens2sents = defaultdict(list)
        assert len(corpus.test.datasets) == 1
        for sent in corpus.test.datasets[0].sentences:
            tokens2sents[tuple(tok.text for tok in sent)].append(sent)

        with open(pred_path, encoding="utf8") as f:
            lines = f.readlines()
        if lines and lines[-1].strip():
            lines.append("\n")  # ensure last sentence is terminated by a blank line

        with tempfile.NamedTemporaryFile("w", buffering=1, encoding="utf8") as f:
            tokens, preds = [], []
            for line in tqdm(lines, desc="Reading predictions"):
                if line.strip():
                    tok, _, pred = line.split()
                    tokens.append(tok)
                    preds.append(pred)
                else:
                    max_f1 = -float("inf")
                    golds = []
                    for sent in tokens2sents[tuple(tokens)]:
                        for token in sent:
                            token.set_label("gold_bio", "O")
                        for true_label in sent.get_labels("evidence?"):
                            prefix = "B-"
                            for token in true_label.data_point:
                                token.set_label(
                                    "gold_bio", f"{prefix}{true_label.value}"
                                )
                                prefix = "I-"
                        with tempfile.NamedTemporaryFile(
                            "w", buffering=1, encoding="utf8"
                        ) as ftmp:
                            for i, token in enumerate(sent):
                                print(
                                    f"{token.text} {token.get_label('gold_bio').value} {preds[i]}",
                                    file=ftmp,
                                )
                            print(file=ftmp)
                            _, _, _, f1 = _run_eval(ftmp.name, is_raw, _log=None)
                        if f1 > max_f1:
                            max_f1 = f1
                            golds = [
                                token.get_label("gold_bio").value for token in sent
                            ]
                    assert len(preds) == len(golds)
                    for cols in zip(tokens, golds, preds):
                        print(" ".join(cols), file=f)
                    print(file=f)
                    tokens, preds = [], []
            values = _run_eval(f.name, is_raw)

    if not (_run is None and _log is None):
        for name, value in zip("accuracy precision recall f1-score".split(), values):
            if _run is not None:
                _run.log_scalar(name, value)
            if _log is not None:
                _log.info("%s: %.1f%%", name, 100 * value)
    return values if return_result else None
