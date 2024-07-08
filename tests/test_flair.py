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

import csv
import json

import flair
import pytest
from flair.data import Dictionary, Sentence
from flair.datasets import ClassificationCorpus
from flair.datasets.sequence_labeling import JsonlCorpus
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.trainers.plugins import MetricRecord, TrainerPlugin


@pytest.fixture
def create_trainer(tmp_path):
    def _write_data_file(set_, texts, labels_list):
        if not isinstance(texts, list):
            texts = [texts]
            labels_list = [labels_list]
        with open(tmp_path / f"{set_}.txt", "w", encoding="utf8") as f:
            for text, labels in zip(texts, labels_list):
                label_part = " ".join(f"__label__{l}" for l in labels)
                print(label_part, text, file=f)

    def _create_trainer(train_data, dev_data=None, test_data=None, corpus_kwargs=None):
        if corpus_kwargs is None:
            corpus_kwargs = {}

        label_dict = Dictionary(add_unk=False)

        def _update_label_dict(labels_list):
            for labels in labels_list:
                if not isinstance(labels, list):
                    labels = [labels]
                for l in labels:
                    label_dict.add_item(l)

        _write_data_file("train", *train_data)
        _update_label_dict(train_data[1])
        if dev_data is not None:
            _write_data_file("dev", *dev_data)
            _update_label_dict(dev_data[1])
        if test_data is not None:
            _write_data_file("test", *test_data)
            _update_label_dict(test_data[1])
        corpus = ClassificationCorpus(
            tmp_path,
            label_type="class",
            train_file="train.txt",
            dev_file=None if dev_data is None else "dev.txt",
            test_file=None if test_data is None else "test.txt",
            **corpus_kwargs,
        )
        doc_embeddings = TransformerDocumentEmbeddings("bert-base-uncased")
        classifier = TextClassifier(
            doc_embeddings,
            label_type="class",
            label_dictionary=label_dict,
            multi_label=True,
        )
        return ModelTrainer(classifier, corpus)

    return _create_trainer


@pytest.mark.slow
def test_trainer_plugin(create_trainer, tmp_path):
    class MyPlugin(TrainerPlugin):
        def __init__(self):
            super().__init__()
            self.count = 0

        @TrainerPlugin.hook
        def metric_recorded(self, record):
            assert isinstance(record, MetricRecord)
            assert all(isinstance(x, str) for x in record.name.parts)
            assert hasattr(record, "value")
            assert isinstance(record.global_step, int)

        @TrainerPlugin.hook
        def after_evaluation(self, *args, **kwargs):
            self.count += 1
            assert len(args) == 0
            assert len(kwargs) == 3
            assert isinstance(kwargs["current_model_is_best"], bool)
            assert isinstance(kwargs["epoch"], int)
            assert len(kwargs["validation_scores"]) == 2  # score, loss
            assert all(isinstance(x, float) for x in kwargs["validation_scores"])

    trainer = create_trainer(("foo", ["A", "B"]), dev_data=("bar", ["A"]))
    plugin = MyPlugin()

    trainer.fine_tune(tmp_path / "artifacts", max_epochs=3, plugins=[plugin])

    assert plugin.count == 3


def test_jsonl_corpus(tmp_path):
    with (tmp_path / "train.jsonl").open("w", encoding="utf8") as f:
        print(
            json.dumps({"data": "foo bar baz", "label": [[0, 2, "A"], [4, 8, "B"]]}),
            file=f,
        )
        print(json.dumps({"data": "quux quux", "label": []}), file=f)

    corpus = JsonlCorpus(tmp_path)

    assert len(corpus.train) == 2
    assert corpus.train[0].text == "foo bar baz"
    spans = corpus.train[0].get_spans()
    assert len(spans) == 2
    assert spans[0].start_position == 0
    assert spans[0].end_position == 3
    assert spans[0].text == "foo"
    assert spans[0].get_label().value == "A"
    assert spans[1].start_position == 4
    assert spans[1].end_position == 7
    assert spans[1].text == "bar"
    assert spans[1].get_label().value == "B"
    assert corpus.train[1].text == "quux quux"
    assert not corpus.train[1].get_spans()


def test_sentence_has_no_spans():
    assert not Sentence("foo bar").get_spans()


@pytest.mark.slow
def test_sequence_tagger_train_and_predict(tmp_path):
    flair.set_seed(0)
    with (tmp_path / "train.jsonl").open("w", encoding="utf8") as f:
        print(
            json.dumps({"data": "foo bar baz", "label": [[0, 2, "A"], [8, 10, "A"]]}),
            file=f,
        )
        print(json.dumps({"data": "quux quux", "label": [[5, 8, "A"]]}), file=f)
    corpus = JsonlCorpus(tmp_path, label_type="tag")
    tag_dict = corpus.make_label_dictionary(label_type="tag")
    tagger = SequenceTagger(TransformerWordEmbeddings(), tag_dict, tag_type="tag")
    trainer = ModelTrainer(tagger, corpus)

    trainer.fine_tune(tmp_path / "artifacts", max_epochs=1)
    tagger = SequenceTagger.load(tmp_path / "artifacts" / "final-model.pt")
    sent = Sentence("foo bar baz quux")
    tagger.predict(sent)

    assert (tmp_path / "artifacts").exists()
    assert sent.get_spans()


def test_sentence_spans_labelled():
    s = Sentence("foo bar")
    s[0:3].add_label("tag", "B-ARG0")

    assert s.get_spans()
    assert s.get_spans()[0].get_label().value == "B-ARG0"


@pytest.mark.slow
def test_sequence_tagger_bio(tmp_path):
    flair.set_seed(0)
    with (tmp_path / "train.jsonl").open("w", encoding="utf8") as f:
        print(
            json.dumps({"data": "foo bar", "label": [[0, len("foo bar") - 1, "A"]]}),
            file=f,
        )
        print(
            json.dumps({"data": "baz", "label": [[0, len("baz") - 1, "A"]]}),
            file=f,
        )
    with (tmp_path / "dev.jsonl").open("w", encoding="utf8") as f:
        print(
            json.dumps({"data": "foo bar", "label": [[0, len("foo bar") - 1, "A"]]}),
            file=f,
        )
        print(
            json.dumps({"data": "baz", "label": [[0, len("baz") - 1, "A"]]}),
            file=f,
        )
    corpus = JsonlCorpus(tmp_path, label_type="tag")
    tag_dict = corpus.make_label_dictionary(label_type="tag")
    tagger = SequenceTagger(TransformerWordEmbeddings(), tag_dict, tag_type="tag")
    trainer = ModelTrainer(tagger, corpus)

    trainer.fine_tune(tmp_path / "artifacts", max_epochs=20)

    assert tagger.tag_format == "BIOES"  # BIOES is used for training
    words, true_tags, pred_tags = [], [], []
    with (tmp_path / "artifacts" / "dev.tsv").open(encoding="utf8") as f:
        for row in csv.reader(f, delimiter=" "):
            if row:
                words.append(row[0])
                true_tags.append(row[1])
                pred_tags.append(row[2])
    assert " ".join(words) == "foo bar baz"
    # BIO is used for representing spans in prediction
    assert true_tags == ["B-A", "I-A", "B-A"]
    assert pred_tags == ["B-A", "I-A", "B-A"]
