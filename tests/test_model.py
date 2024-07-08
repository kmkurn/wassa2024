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

import textwrap

import flair
import pytest
from run_span_predictor import train as train_span


@pytest.mark.slow
@pytest.mark.parametrize("method", ["ReL", "MV"])
def test_span_predictor_train(write_span_prediction_data, data_dir, tmp_path, method):
    write_span_prediction_data()

    train_span(str(data_dir), str(tmp_path / "artifacts"), method, max_epochs=1)

    assert (tmp_path / "artifacts" / "final-model.pt").exists()


@pytest.mark.slow
@pytest.mark.parametrize("method", ["ReL", "MV"])
def test_span_predictor_train_eval_sets_exist(
    write_span_prediction_data, data_dir, tmp_path, method
):
    write_span_prediction_data()
    write_span_prediction_data(
        [("foo", "A", []), ("bar baz", "A", [(0, 3), (4, 3)])], split_name="dev"
    )
    write_span_prediction_data(
        [("foo", "A", []), ("bar baz", "A", [(0, 3), (4, 3)])], split_name="test"
    )
    artifacts_dir = tmp_path / "artifacts"

    train_span(str(data_dir), str(artifacts_dir), method, max_epochs=1)
    dev_lines = (artifacts_dir / "dev.tsv").read_text("utf8").splitlines()
    test_lines = (artifacts_dir / "test.tsv").read_text("utf8").splitlines()
    expected = [
        ["foo O", "< O", "sep O", "> O", "A O"],
        ["bar B-Yes", "baz B-Yes", "< O", "sep O", "> O", "A O"],
    ]

    for lines in (dev_lines, test_lines):
        i = j = cnt = 0
        for line in lines:
            line = line.rstrip()
            if not line:
                i += 1
                j = 0
                continue
            cols = line.split()
            assert len(cols) == 3
            assert cols[:2] == expected[i][j].split()
            j += 1
            cnt += 1
        assert cnt


@pytest.mark.slow
@pytest.mark.parametrize("method", ["ReL", "MV"])
def test_span_predictor_skip_saving_model(
    write_span_prediction_data, data_dir, tmp_path, method
):
    write_span_prediction_data()
    artifacts_dir = tmp_path / "artifacts"

    train_span(
        str(data_dir), str(artifacts_dir), method, max_epochs=1, save_model=False
    )

    assert not (artifacts_dir / "final-model.pt").exists()


@pytest.mark.slow
def test_span_predictor_mv(write_span_prediction_data, data_dir, tmp_path):
    flair.set_seed(0)
    write_span_prediction_data(
        [
            ("foo bar baz", "A", [(0, 7)]),
            ("foo bar baz", "B", [(0, 3)]),
            ("foo bar baz", "A", [(4, 7)]),
        ]
    )
    write_span_prediction_data([("foo bar baz", "A", [(4, 3)])], split_name="test")

    # train to overfit
    train_span(str(data_dir), str(tmp_path / "artifacts"), method="MV", max_epochs=100)
    expected = textwrap.dedent(
        """
    foo O O
    bar B-Yes B-Yes
    baz O O
    < O O
    sep O O
    > O O
    A O O

    """
    ).lstrip()

    assert (tmp_path / "artifacts" / "test.tsv").read_text("utf8") == expected


def test_span_predictor_random(write_span_prediction_data, data_dir, tmp_path):
    write_span_prediction_data(
        [("foo bar", "A", [(4, 3)]), ("foo bar baz quux", "B", [(4, 7)])],
        split_name="test",
    )
    expected = [
        "foo O",
        "bar B-Yes",
        "< O",
        "sep O",
        "> O",
        "A O",
        "foo O",
        "bar B-Yes",
        "baz I-Yes",
        "quux O",
        "< O",
        "sep O",
        "> O",
        "B O",
    ]

    train_span(str(data_dir), str(tmp_path / "artifacts"), method="Random")

    i = 0
    for line in (tmp_path / "artifacts" / "test.tsv").read_text("utf8").splitlines():
        if not line:
            continue
        splits = line.split()
        assert splits[:2] == expected[i].split()
        assert splits[2] in ("B-Yes", "I-Yes", "O")
        i += 1
