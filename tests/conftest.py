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
from pathlib import Path
from typing import Optional, Sequence

import pytest


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def write_span_prediction_data(data_dir):
    def _write_data(
        data: Optional[Sequence[tuple[str, str, Sequence[tuple[int, int]]]]] = None,
        split_name: str = "train",
        annotators: Optional[Sequence[str]] = None,
    ) -> Path:
        if data is None:
            data = [("foo bar\nbaz", "A", [(0, 2)]), ("foo bar\nbaz", "B", [(8, 3)])]
        if annotators is None:
            annotators = [str(i) for i in range(len(data))]
        with (data_dir / f"{split_name}.jsonl").open("w", encoding="utf8") as f:
            for (text, label, spans), ann in zip(data, annotators):
                print(
                    json.dumps(
                        {
                            "input": {"text": text, "label": label},
                            "output": {"spans": spans},
                            "metadata": {"annotator": ann},
                        }
                    ),
                    file=f,
                )
        return data_dir

    return _write_data
