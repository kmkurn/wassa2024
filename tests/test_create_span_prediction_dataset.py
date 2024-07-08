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

import functools
import random
from typing import Iterator, Iterable

import pandas as pd
import pytest
from create_span_prediction_dataset import df2examples, split_kfold


def test_train_split_only(tmp_path):
    df = pd.DataFrame(
        {
            "Sample_Text": [
                "foo\nbar",
                "foo\nbar",
                "foo\nbar",
                "bar",
                "bar",
                "bar",
                "baz",
                "baz",
            ],
            "Type": [
                "highlight",
                "overview",
                "highlight",
                "highlight",
                "highlight",
                "highlight",
                "highlight",
                "highlight",
            ],
            "Area_of_Law": ["A", "Z", "B", "B", "A", "A", "C C", "C C"],
            "User_ID": [1, 1, 1, 1, 2, 1, 3, 3],
            "Tagged_Text": ["foo", None, "bar", "ba", "ar", "r", "b", "az"],
            "Tagged_Text_Start": [0, None, 4, 0, 1, 2, 0, 1],
        }
    )
    df = df.convert_dtypes().sample(frac=1.0)
    expected = [
        {
            "input": {"text": "foo\nbar", "label": "A"},
            "output": {"spans": [(0, 3)]},
            "metadata": {"annotator": 1},
        },
        {
            "input": {"text": "foo\nbar", "label": "B"},
            "output": {"spans": [(4, 3)]},
            "metadata": {"annotator": 1},
        },
        {
            "input": {"text": "bar", "label": "B"},
            "output": {"spans": [(0, 2)]},
            "metadata": {"annotator": 1},
        },
        {
            "input": {"text": "bar", "label": "A"},
            "output": {"spans": [(1, 2)]},
            "metadata": {"annotator": 2},
        },
        {
            "input": {"text": "bar", "label": "A"},
            "output": {"spans": [(2, 1)]},
            "metadata": {"annotator": 1},
        },
        {
            "input": {"text": "baz", "label": "C C"},
            "output": {"spans": [(0, 1), (1, 2)]},
            "metadata": {"annotator": 3},
        },
    ]

    examples = list(df2examples(df))

    for obj in expected:
        assert obj in examples
        examples.remove(obj)
    assert not examples


@pytest.fixture
def create_df():
    def _create_df(n_unique_text_label_pairs: int, size: int) -> pd.DataFrame:
        N, K = n_unique_text_label_pairs, size
        unique_texts = [" ".join(["foo"] * (n + 1)) for n in range(N)]
        unique_labels = [f"L{n}" for n in range(N)]
        txt_lab_pairs = random.choices(list(zip(unique_texts, unique_labels)), k=K)
        texts, labels = zip(*txt_lab_pairs)
        df = pd.DataFrame(
            {
                "Sample_Text": texts,
                "Area_of_Law": labels,
                "User_ID": random.choices(list(range(5)), k=K),
            }
        )
        df["Type"] = "highlight"
        df["Tagged_Text"] = "foo"
        df["Tagged_Text_Start"] = 0
        return df

    return _create_df


@pytest.fixture
def get_all_fold_text_label_pairs():
    def _get_all_fold_text_label_pairs(
        all_fold_examples: Iterable[Iterable[dict[str, dict]]], cast=set
    ) -> list[set[tuple[str, str]]]:
        return [cast(_get_text_label_pairs(examples)) for examples in all_fold_examples]

    def _get_text_label_pairs(examples) -> Iterator[tuple[str, str]]:
        for ex in examples:
            yield ex["input"]["text"], ex["input"]["label"]

    return _get_all_fold_text_label_pairs


def test_kfold_test_sets_are_partitions(create_df, get_all_fold_text_label_pairs):
    df = create_df(n_unique_text_label_pairs=10, size=100)
    n_folds = 3

    _, all_fold_tst_examples = split_kfold(list(df2examples(df)), n_folds)

    tst_text_label_pairs_ls = get_all_fold_text_label_pairs(all_fold_tst_examples)
    for i in range(len(tst_text_label_pairs_ls)):
        for j in range(i + 1, len(tst_text_label_pairs_ls)):
            assert not (tst_text_label_pairs_ls[i] & tst_text_label_pairs_ls[j])
    assert functools.reduce(lambda x, y: x | y, tst_text_label_pairs_ls) == set(
        zip(df.Sample_Text, df.Area_of_Law)
    )


def test_kfold_equally_sized_test_sets(create_df, get_all_fold_text_label_pairs):
    n_unique_text_label_pairs = 10
    df = create_df(n_unique_text_label_pairs, size=100)
    n_folds = 3

    _, all_fold_tst_examples = split_kfold(list(df2examples(df)), n_folds)

    tst_text_label_pairs_ls = get_all_fold_text_label_pairs(all_fold_tst_examples)

    for pairs in tst_text_label_pairs_ls:
        assert len(pairs) - (n_unique_text_label_pairs // n_folds) in (0, 1)


def test_kfold_train_test_sets_are_partitions(create_df, get_all_fold_text_label_pairs):
    df = create_df(n_unique_text_label_pairs=10, size=100)
    n_folds = 3

    all_fold_trn_examples, all_fold_tst_examples = split_kfold(
        list(df2examples(df)), n_folds
    )

    trn_text_label_pairs_ls = get_all_fold_text_label_pairs(all_fold_trn_examples)
    tst_text_label_pairs_ls = get_all_fold_text_label_pairs(all_fold_tst_examples)
    all_text_label_pairs = set(zip(df.Sample_Text, df.Area_of_Law))
    for trn_text_label_pairs, tst_text_label_pairs in zip(
        trn_text_label_pairs_ls, tst_text_label_pairs_ls
    ):
        assert trn_text_label_pairs | tst_text_label_pairs == all_text_label_pairs
        assert not (trn_text_label_pairs & tst_text_label_pairs)


def test_kfold_with_dev(create_df, get_all_fold_text_label_pairs):
    df = create_df(n_unique_text_label_pairs=1000, size=10_000)
    n_folds = 3

    all_fold_split_examples = split_kfold(list(df2examples(df)), n_folds, dev=0.1)

    split_text_label_pairs_ls = [
        get_all_fold_text_label_pairs(all_fold_examples)
        for all_fold_examples in all_fold_split_examples
    ]

    for trn_text_label_pairs, dev_text_label_pairs, tst_text_label_pairs in zip(
        *split_text_label_pairs_ls
    ):
        assert (
            trn_text_label_pairs | dev_text_label_pairs | tst_text_label_pairs
        ) == set(zip(df.Sample_Text, df.Area_of_Law))
        assert not (dev_text_label_pairs & trn_text_label_pairs)
        assert not (dev_text_label_pairs & tst_text_label_pairs)
        assert len(dev_text_label_pairs) / (
            len(dev_text_label_pairs) + len(trn_text_label_pairs)
        ) == pytest.approx(0.1, abs=1e-3)


def test_kfold_with_shuffling(create_df, get_all_fold_text_label_pairs):
    df = create_df(n_unique_text_label_pairs=10, size=10)
    random.seed(0)

    _, all_fold_tst_examples = split_kfold(list(df2examples(df)), fold=2, shuffle=True)
    _, all_fold_tst_examples2 = split_kfold(list(df2examples(df)), fold=2, shuffle=True)

    assert all_fold_tst_examples != all_fold_tst_examples2


def test_kfold_majority_voted_spans_in_test_sets():
    df = pd.DataFrame(
        {
            "Sample_Text": [
                "foo",
                "foo",
                "bar quux",
                "bar quux",
                "bar quux",
                "bar quux",
            ],
            "User_ID": [1, 2, 1, 2, 1, 2],
            "Tagged_Text": ["f", "o", "ba", "ar q", "qu", "ux"],
            "Tagged_Text_Start": [0, 1, 0, 1, 4, 6],
        }
    )
    df["Area_of_Law"] = "A"
    df["Type"] = "highlight"

    _, all_fold_tst_examples = split_kfold(list(df2examples(df)), fold=2)

    assert len(all_fold_tst_examples[0]) == 1
    test_example_1 = all_fold_tst_examples[0][0]
    assert len(all_fold_tst_examples[1]) == 1
    test_example_2 = all_fold_tst_examples[1][0]
    assert {
        "input": {"text": "bar quux", "label": "A"},
        "output": {"spans": [(1, 1), (4, 1)]},
    } in (test_example_1, test_example_2)
    assert {"input": {"text": "foo", "label": "A"}, "output": {"spans": []}} in (
        test_example_1,
        test_example_2,
    )


def test_kfold_majority_voted_spans_in_dev_sets():
    random.seed(0)
    df = pd.DataFrame(
        {
            "Sample_Text": [
                "foo bar",
                "foo bar",
                "foo bar",
                "bar baz",
                "bar baz",
                "bar baz",
                "baz quux",
                "baz quux",
                "baz quux",
                "baz quux",
            ],
            "User_ID": [1, 2, 1, 1, 2, 1, 1, 2, 1, 2],
            "Tagged_Text": [
                "foo",
                "ba",
                "r",
                "bar",
                "bar",
                "baz",
                "baz q",
                "z q",
                "ux",
                "uux",
            ],
            "Tagged_Text_Start": [0, 4, 6, 0, 0, 4, 0, 2, 6, 5],
        }
    )
    df["Area_of_Law"] = "A"
    df["Type"] = "highlight"
    expected = [
        {
            "input": {"text": "foo bar", "label": "A"},
            "output": {"spans": []},
        },
        {
            "input": {"text": "bar baz", "label": "A"},
            "output": {"spans": [(0, 3)]},
        },
        {
            "input": {"text": "baz quux", "label": "A"},
            "output": {"spans": [(2, 3), (6, 2)]},
        },
    ]

    _, all_fold_dev_examples, _ = split_kfold(list(df2examples(df)), fold=3, dev=0.5)

    assert all(len(exs) == 1 for exs in all_fold_dev_examples)
    dev_examples = [exs[0] for exs in all_fold_dev_examples]
    for ex in dev_examples:
        assert ex in expected
