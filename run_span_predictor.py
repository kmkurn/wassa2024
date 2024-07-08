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
import tempfile
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
from flair.datasets.sequence_labeling import JsonlCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.trainers.plugins import MetricRecord, TrainerPlugin
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.run import Run

from create_span_prediction_dataset import aggregate_spans

ex = Experiment("span-prediction")
if "SACRED_MONGO_URL" in os.environ:
    ex.observers.append(
        MongoObserver(
            url=os.environ["SACRED_MONGO_URL"],
            db_name=os.getenv("SACRED_DB_NAME", "sacred"),
        )
    )


@ex.config
def default():
    # directory containing {train,dev,test}.jsonl
    data_dir = "data"
    # directory to save training artifacts
    artifacts_dir = "artifacts"
    # method [Random, ReL, MV]
    method = "ReL"
    # max number of epochs
    max_epochs = 10
    # text to separate input text and label
    sep_text = "<sep>"
    # batch size
    batch_size = 4
    # learning rate
    lr = 5e-5
    # whether to save the trained model
    save_model = True
    # name of HF pretrained LM as base
    model_name = "bert-base-uncased"
    # if given, split a batch into chunks for grad accumulation
    batch_chunk_size = None


@ex.named_config
def deberta_v3_base():
    model_name = "microsoft/deberta-v3-base"


class SacredLogMetricsPlugin(TrainerPlugin):
    def __init__(self, run: Run) -> None:
        super().__init__()
        self.__run = run

    @staticmethod
    def __should_log(record: MetricRecord) -> bool:
        is_batch_metric = len(record.name.parts) >= 2 and record.name.parts[1] in (
            "batch_loss",
            "gradient_norm",
        )
        return record.is_scalar and (
            not is_batch_metric or record.global_step % 100 == 0
        )

    @TrainerPlugin.hook
    def metric_recorded(self, record: MetricRecord) -> None:
        if self.__should_log(record):
            try:
                value = record.value.item()
            except AttributeError:
                value = record.value
            self.__run.log_scalar(str(record.name), value, record.global_step)


def preprocess(
    text: str, label: str, separator: str = "<sep>"
) -> str:
    return f"{text} {separator} {label}"


def read_jsonl(path: Path, sep_text: str = "<sep>") -> Iterator[dict]:
    with path.open(encoding="utf8") as f:
        for line in f:
            ex = json.loads(line)
            if sep_text in ex["input"]["text"]:
                warnings.warn("Input text contains separator text")
            if sep_text in ex["input"]["label"]:
                warnings.warn("Input label contains separator text")
            yield {
                "text": preprocess(ex["input"]["text"], ex["input"]["label"]),
                "spans": [
                    [start, start + length, "Yes"]
                    for start, length in ex["output"]["spans"]
                    if ex["input"]["text"][
                        start : start + length
                    ].strip()  # Flair raises error without this
                ],
            }


@ex.automain
def train(
    data_dir,
    artifacts_dir,
    method="ReL",
    max_epochs=10,
    batch_size=4,
    lr=5e-5,
    sep_text="<sep>",
    save_model=True,
    model_name="bert-base-uncased",
    batch_chunk_size=None,
    _run=None,
    _log=None,
    _rnd=None,
):
    """Train a span prediction model."""
    data_dir = Path(data_dir)
    if method == "ReL":
        data = list(read_jsonl(data_dir / "train.jsonl", sep_text))
    elif method == "MV":
        with (data_dir / "train.jsonl").open(encoding="utf8") as f:
            examples = [json.loads(l) for l in f]
        if _log is not None:
            _log.info("Read %d training examples", len(examples))
        data = []
        for ex in aggregate_spans(examples):
            if sep_text in ex["input"]["text"]:
                warnings.warn("Input text contains separator text")
            if sep_text in ex["input"]["label"]:
                warnings.warn("Input label contains separator text")
            data.append(
                {
                    "text": preprocess(ex["input"]["text"], ex["input"]["label"]),
                    "spans": [
                        [start, start + length, "Yes"]
                        for start, length in ex["output"]["spans"]
                        if ex["input"]["text"][
                            start : start + length
                        ].strip()  # Flair raises error without this
                    ],
                }
            )
        if _log is not None:
            _log.info("%d training examples remain after MV", len(data))
    elif method == "Random":
        data = []  # no training
    else:
        raise ValueError(f"unrecognised method: {method}")
    dev_data, test_data = [], []
    if (data_dir / "dev.jsonl").exists():
        dev_data.extend(read_jsonl(data_dir / "dev.jsonl", sep_text))
    if (data_dir / "test.jsonl").exists():
        test_data.extend(read_jsonl(data_dir / "test.jsonl", sep_text))

    with tempfile.TemporaryDirectory() as tmpdirname:
        with (Path(tmpdirname) / "train.jsonl").open("w", encoding="utf8") as f:
            for dat in data:
                print(json.dumps(dat), file=f)
        if dev_data:
            with (Path(tmpdirname) / "dev.jsonl").open("w", encoding="utf8") as f:
                for dat in dev_data:
                    print(json.dumps(dat), file=f)
        if test_data:
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

    assert corpus.train is not None

    if method == "Random":
        if _rnd is None:
            _rnd = np.random.default_rng()
        Path(artifacts_dir).mkdir()
        if corpus.test is not None:
            assert corpus.test.datasets
            with (Path(artifacts_dir) / "test.tsv").open("w", encoding="utf8") as f:
                for sent in corpus.test.datasets[0].sentences:
                    for token in sent:
                        token.set_label("true_bio", "O")
                        token.set_label(
                            "pred_bio", _rnd.choice(["B-Yes", "I-Yes", "O"])
                        )
                    for true_label in sent.get_labels("evidence?"):
                        prefix = "B-"
                        for token in true_label.data_point:
                            token.set_label("true_bio", f"{prefix}{true_label.value}")
                            prefix = "I-"
                    for token in sent:
                        print(
                            f"{token.text} {token.get_label('true_bio').value} {token.get_label('pred_bio').value}",
                            file=f,
                        )
                    print(file=f)
        return

    tagger = SequenceTagger(
        TransformerWordEmbeddings(model_name),
        corpus.make_label_dictionary(label_type="evidence?"),
        tag_type="evidence?",
    )
    trainer = ModelTrainer(tagger, corpus)
    plugins = []
    if _run is not None:
        plugins.append(SacredLogMetricsPlugin(_run))
    trainer.fine_tune(
        artifacts_dir,
        learning_rate=lr,
        mini_batch_size=batch_size,
        mini_batch_chunk_size=batch_chunk_size,
        max_epochs=max_epochs,
        plugins=plugins,
    )
    if not save_model:
        (Path(artifacts_dir) / "final-model.pt").unlink()
