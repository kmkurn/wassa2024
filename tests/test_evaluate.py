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
import textwrap

import pytest
from run_span_eval import evaluate as eval_span


@pytest.mark.parametrize("is_raw", [False, True])
def test_eval_span_test_preds(is_raw, tmp_path):
    (tmp_path / "test.tsv").write_text(
        textwrap.dedent(
            """
    foo B-Yes O
    bar I-Yes B-Yes

    bar B-Yes B-Yes
    baz O O

    foo O O
    bar B-Yes B-Yes
    baz I-Yes I-Yes

    """
        ).lstrip(),
        "utf8",
    )

    res = eval_span(str(tmp_path / "test.tsv"), is_raw)

    assert len(res) == 4
    assert all(isinstance(r, float) for r in res)
    assert all(0 <= r <= 1 for r in res)


@pytest.mark.parametrize("is_raw", [False, True])
def test_eval_span_test_preds_sep_token(is_raw, tmp_path):
    (tmp_path / "testa.tsv").write_text(
        textwrap.dedent(
            """
    bar B-Yes B-Yes
    < O I-Yes
    se O O
    p O O
    > O B-Yes
    baz O I-Yes

    foo O O
    bar B-Yes B-Yes
    <sep> O B-Yes
    baz B-Yes I-Yes

    """
        ).lstrip(),
        "utf8",
    )
    (tmp_path / "testb.tsv").write_text(
        textwrap.dedent(
            """
    bar B-Yes B-Yes

    foo O O
    bar B-Yes B-Yes

    """
        ),
        "utf8",
    )

    assert eval_span(str(tmp_path / "testa.tsv"), is_raw) == pytest.approx(
        eval_span(str(tmp_path / "testb.tsv"), is_raw)
    )


def test_eval_span_raw(tmp_path):
    (tmp_path / "testa.tsv").write_text(
        textwrap.dedent(
            """
    foo B-Yes O
    bar I-Yes B-Yes

    bar B-Yes B-Yes
    baz O O

    """
        ).lstrip(),
        "utf8",
    )
    (tmp_path / "testb.tsv").write_text(
        textwrap.dedent(
            """
    foo Yes O
    bar Yes Yes

    bar Yes Yes
    baz O O

    """
        ).lstrip(),
        "utf8",
    )

    assert eval_span(str(tmp_path / "testa.tsv"), is_raw=True) == pytest.approx(
        eval_span(str(tmp_path / "testb.tsv"), is_raw=True)
    )


def test_eval_span_multi_references(tmp_path):
    (tmp_path / "testa.tsv").write_text(
        textwrap.dedent(
            """
    foo O B-Yes
    bar O I-Yes
    bar O O
    baz O B-Yes
    < O O
    sep O O
    > O O
    A O O

    """
        ).lstrip(),
        "utf8",
    )
    with (tmp_path / "disagg_test.jsonl").open("w", encoding="utf8") as f:
        print(
            json.dumps(
                {
                    "input": {"text": "foo bar bar baz", "label": "A"},
                    "output": {"spans": [(0, 3), (12, 3)]},
                    "metadata": {"annotator": "2"},
                }
            ),
            file=f,
        )
        print(
            json.dumps(
                {
                    "input": {"text": "foo bar bar baz", "label": "A"},
                    "output": {"spans": [(0, 7)]},
                    "metadata": {"annotator": "1"},
                }
            ),
            file=f,
        )
    (tmp_path / "testb.tsv").write_text(
        textwrap.dedent(
            """
    foo B-Yes B-Yes
    bar I-Yes I-Yes
    bar O O
    baz O B-Yes
    < O O
    sep O O
    > O O
    A O O

    """
        ).lstrip(),
        "utf8",
    )

    assert eval_span(
        str(tmp_path / "testa.tsv"),
        disagg_test_path=str(tmp_path / "disagg_test.jsonl"),
    ) == pytest.approx(eval_span(str(tmp_path / "testb.tsv")))
