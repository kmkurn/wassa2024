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

import argparse
import json
import random
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

Example = dict[str, dict]


def df2examples(df: pd.DataFrame, /) -> Iterator[Example]:
    for (uid, text, label), spans in (  # type: ignore[misc]
        df[df.Type == "highlight"]
        .groupby(["User_ID", "Sample_Text", "Area_of_Law"])
        .apply(_get_spans)
        .items()
    ):
        yield {
            "input": {"text": text, "label": label},  # type: ignore[has-type]
            "output": {"spans": spans},
            "metadata": {"annotator": uid},  # type: ignore[has-type]
        }


def split_kfold(
    examples: list[Example],
    fold: int,
    dev: float = 0.0,
    shuffle: bool = False,
    seed: int = 0,
) -> (
    tuple[list[list[Example]], list[list[Example]]]
    | tuple[list[list[Example]], list[list[Example]], list[list[Example]]]
):
    if shuffle:
        random.shuffle(examples)
    pair2id: dict = {}
    for ex in examples:
        pair = (ex["input"]["text"], ex["input"]["label"])
        if pair not in pair2id:
            pair2id[pair] = len(pair2id)

    all_fold_trn_examples, all_fold_dev_examples, all_fold_tst_examples = [], [], []
    for trn_indices, tst_indices in GroupKFold(fold).split(
        examples,
        groups=[pair2id[ex["input"]["text"], ex["input"]["label"]] for ex in examples],
    ):
        tst_examples = [examples[i] for i in tst_indices]
        all_fold_tst_examples.append(list(aggregate_spans(tst_examples)))
        if dev:
            pair_ids = [
                pair2id[examples[i]["input"]["text"], examples[i]["input"]["label"]]
                for i in trn_indices
            ]
            [(trn_idxs, dev_idxs)] = list(
                GroupShuffleSplit(n_splits=1, test_size=dev, random_state=seed).split(
                    trn_indices, groups=pair_ids
                )
            )
            dev_examples = [examples[trn_indices[i]] for i in dev_idxs]
            all_fold_dev_examples.append(list(aggregate_spans(dev_examples)))
            trn_indices = [trn_indices[i] for i in trn_idxs]
        all_fold_trn_examples.append([examples[i] for i in trn_indices])

    if all_fold_dev_examples:
        return all_fold_trn_examples, all_fold_dev_examples, all_fold_tst_examples
    return all_fold_trn_examples, all_fold_tst_examples


def aggregate_spans(examples: list[Example]) -> Iterator[Example]:
    pair2spans = defaultdict(list)
    pair2annotators = defaultdict(set)
    for ex in examples:
        pair2spans[ex["input"]["text"], ex["input"]["label"]].extend(
            ex["output"]["spans"]
        )
        pair2annotators[ex["input"]["text"], ex["input"]["label"]].add(
            ex["metadata"]["annotator"]
        )
    pair2counter = {
        pair: _compute_span_counter(spans) for pair, spans in pair2spans.items()
    }
    for (text, label), counter in pair2counter.items():
        yield {
            "input": {"text": text, "label": label},
            "output": {
                "spans": sorted(
                    [
                        span
                        for span, cnt in counter.items()
                        if cnt > len(pair2annotators[text, label]) // 2
                    ],
                    key=lambda span: span.start,
                )
            },
        }


def _compute_span_counter(spans: Iterable["Span"]) -> Counter["Span"]:
    positions = [i for start, length in spans for i in range(start, start + length)]
    if not positions:
        return Counter()

    min_pos, max_pos = min(positions), max(positions)
    pos2count = Counter(positions)
    span2count: Counter["Span"] = Counter()
    start = min_pos
    for end in range(min_pos + 1, max_pos + 2):
        if pos2count[end] != pos2count[start]:
            assert all(pos2count[i] == pos2count[start] for i in range(start + 1, end))
            span2count[Span(start, end - start)] = pos2count[start]
            start = end
    assert end == max_pos + 1 and start == end
    return span2count


def _get_spans(df: pd.DataFrame) -> list["Span"]:
    return [Span(row.Tagged_Text_Start, len(row.Tagged_Text)) for row in df.sort_values("Tagged_Text_Start").itertuples()]  # type: ignore[attr-defined]


class Span(NamedTuple):
    start: int
    length: int


def write_split_to_jsonl(
    split: list[list[Example]],
    output_dir: Path,
    split_name: str,
    encoding: str = "utf8",
) -> None:
    for i, fold_split in enumerate(split):
        with (output_dir / str(i) / f"{split_name}.jsonl").open(
            "w", encoding=encoding
        ) as f:
            for ex in fold_split:
                ex["output"]["spans"] = [
                    Span(int(span.start), span.length) for span in ex["output"]["spans"]
                ]
                print(json.dumps(ex), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """
        Create span prediction dataset in JSONL format from a DataFrame.
        When --fold K is given, labels in test set (and dev sets if --dev is also given)
        are aggregated using majority voting."""
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data", type=Path, help="path to the DataFrame saved as .pkl file"
    )
    parser.add_argument("output_dir", type=Path, help="output directory")
    parser.add_argument(
        "-k",
        "--fold",
        metavar="K",
        default=0,
        type=int,
        help="do K-fold cross-validation split",
    )
    parser.add_argument(
        "--dev",
        default=0.0,
        type=float,
        help="proportion of dev set in non-testing portion of a fold",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="shuffle before splitting"
    )
    parser.add_argument("--encoding", default="utf8", help="file encoding")
    args = parser.parse_args()
    if not args.fold:
        with (args.output_dir / "all.jsonl").open("w", encoding=args.encoding) as f:
            for ex in df2examples(pd.read_pickle(args.data)):
                ex["output"]["spans"] = [
                    Span(int(span.start), span.length) for span in ex["output"]["spans"]
                ]
                print(json.dumps(ex), file=f)
        sys.exit(0)

    for i in range(args.fold):
        (args.output_dir / str(i)).mkdir()
    splits = split_kfold(
        list(df2examples(pd.read_pickle(args.data))), args.fold, args.dev, args.shuffle
    )
    assert len(splits) in (2, 3)
    write_split_to_jsonl(splits[0], args.output_dir, "train", args.encoding)
    write_split_to_jsonl(splits[-1], args.output_dir, "test", args.encoding)
    if len(splits) == 3:
        write_split_to_jsonl(splits[1], args.output_dir, "dev", args.encoding)
