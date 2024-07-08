# Span Prediction Model

## Requirements

1. Python 3.10
1. CUDA 11.7
1. `pip install -r requirements.txt`
1. [CoNLL evaluation script](https://www.cnts.ua.ac.be/conll2000/chunking/conlleval.txt) saved as `conlleval` in the project directory

## Data

### Format

The disaggregated span annotation data must be stored in a [DataFrame](https://pandas.pydata.org/docs/reference/frame.html#dataframe) with at least 6 columns:

1. `Sample_Text`, which is the problem description text;
1. `Type`, which must always equal the string `"highlight"` (otherwise the row will be ignored);
1. `Area_of_Law`, which is an area of law assigned to the problem description;
1. `User_ID`, which is the unique identifier of the expert annotator;
1. `Tagged_Text`, which is the annotated span of text in the problem description; and
1. `Tagged_Text_Start`, which is the start index (0-based) of the annotated span in the problem description.

An example DataFrame is

    pd.DataFrame(
        {
            "Sample_Text": [
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
                "highlight",
                "highlight",
                "highlight",
                "highlight",
                "highlight",
                "highlight",
            ],
            "Area_of_Law": ["A", "B", "B", "A", "A", "C C", "C C"],
            "User_ID": [1, 1, 1, 2, 1, 3, 3],
            "Tagged_Text": ["foo", "bar", "ba", "ar", "r", "b", "az"],
            "Tagged_Text_Start": [0, 4, 0, 1, 2, 0, 1],
        }
    )

where it contains the span annotations of 3 problem descriptions. The first description is

*foo*
<br>
*bar*

which was assigned to areas of law "A" and "B" by Expert 1. The expert also annotated the words *foo* and *bar* to support their areas of law assignment respectively. The second description is *bar* assigned to areas of law:

1. "A" and "B" by Expert 1, who annotated the character *r* and the span of characters *ba* to support their assignments respectively; and
1. "A" by Expert 2, who annotated *ar* to support their assignment.

The third description is *baz* assigned to area of law "C C" by Expert 3, who annotated spans *b* and *az* to support the assignment.

### Creating the dataset

To create a 20-fold cross-validated dataset, run

    ./create_span_prediction_dataset.py -k 20 --dev 0.1 --shuffle /path/to/dataframe.pkl /path/to/output/dir

which shuffles the data before splitting and also samples 10% of the training portion of each fold to create a development set. The script creates 20 directories named 0, 1, and so on under the output directory, each corresponds to a fold. Under each directory, there are `{train,dev,test}.jsonl` files containing the span-annotated examples.

The gold spans in `{dev,test}.jsonl` are majority-voted. To allow evaluation with the best-matched spans, get all the disaggregated annotations with

    ./create_span_prediction_dataset.py /path/to/dataframe.pkl /path/to/output/dir

which saves them in `/path/to/output/dir/all.jsonl` file.

## Training

To train the BERT-based span predictor model, run

    ./run_span_predictor.py with data_dir=/path/to/fold/dir method=<method>

where `/path/to/fold/dir` contains the `{train,dev,test}.jsonl` files, and `<method>` is `Random`, `ReL`, or `MV` for the random baseline, repeated labelling, and majority voting methods respectively.

To train the DeBERTaV3-based model, run

    ./run_span_predictor.py with deberta_v3_base data_dir=/path/to/fold/dir method=<method>

By default, the command saves the trained models, predictions, etc. in `artifacts` directory. Run

    ./run_span_predictor.py print_config

to see other configurable settings.

## CoNLL evaluation

To perform evaluation with the `conlleval` script, run

    ./run_span_eval.py with pred_path=/path/to/artifacts/dir/test.tsv

By default, the script will run span-level evaluation against the majority-voted gold spans. To run word-level evaluation instead, pass `is_raw=True` as an argument:

    ./run_span_eval.py with pred_path=/path/to/artifacts/dir/test.tsv is_raw=True

To evaluate against the best-matched gold spans, specify the disaggregated span annotations as `disagg_test_path`:

    ./run_span_eval.py with pred_path=/path/to/artifacts/dir/test.tsv disagg_test_path=/path/to/output/dir/all.jsonl

## MongoDB integration with Sacred

Both `run_span_predictor.py` and `run_span_eval.py` scripts use [Sacred](https://pypi.org/project/sacred/) and have its `MongoObserver` activated by default. Set `SACRED_MONGO_URL` (and optionally `SACRED_DB_NAME`) environment variable(s) to write experiment runs to a MongoDB instance. For example, set `SACRED_MONGO_URL=mongodb://localhost:27017` if the MongoDB instance is listening on port 27017 on the local machine.

## Running tests

Tests can be run with the command `pytest`. By default, slow tests are excluded. To run them, pass `-m slow` as an argument.
