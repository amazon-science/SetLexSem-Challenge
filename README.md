# Overview

This research repository maintains the code and the results for the research paper: SETLEXSEM CHALLENGE: Using Set Operations to Evaluate the Lexical and Semantic Robustness of Language Models.

_"Set theory has become the standard foundation for mathematics, as every mathematical object can be viewed as a set."_ -[Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/set-theory/)

# Install Package from PyPI and Use
To install the package, please run:
```pip install setlexsem```
You can generate the dataset by:
```python
from setlexsem.generate.generate_sets import make_sets
make_sets(
    set_types=["numbers"],
    n=[10],
    m=[1,2],
    item_len=[1],
    decile_group=None,
    swap_status=None,
    overlap_fraction=[None],
    seed_value=292,
    number_of_data_points= 3
)
# --- or, this can be config based:
config = {
    "set_types": ["numbers"],
    "n": [10],
    "m": [10],
    "item_len": [2],
    "decile_group": None,
    "swap_status": None,
    "overlap_fraction": [0.5],
}
out_data = make_sets(
    config=config,
    number_of_data_points= 5,
    seed_value=292
)
```
You can generate the prompts by:
```python
from setlexsem.generate.generate_prompts import generate_prompts

generate_prompts(
    # data config
    set_types=["numbers"],
    n=[10],
    m=[1,2],
    item_len=[1,3],
    decile_group=None,
    swap_status=None,
    overlap_fraction=[None],
    number_of_data_points= 5,
    random_seed_value=292,
    # prompt config
    op_list=["union", "intersection"],
    k_shot=[0, 1, 3],
    prompt_type=["formal_language"],
    prompt_approach=["baseline", "chain_of_thought"],
    is_fix_shot=[True],
    add_roles=False)
```



# Install

When installing, it's important to upgrade to the most recent pip. This ensures that `setup.py` runs correctly. An outdated version of pip can fail to run the `InstallNltkWordnetAfterPackages` command class in setup.py and cause subsequent errors.

``` bash
/usr/bin/python3 -mvenv venv
. venv/bin/activate
python3 -m pip install --upgrade pip
pip install -e .
pip install -e ."[dev, test]"
```

# NLTK words

If you get errors from `nltk` about package `words` not being installed while
executing the code in this repository, run:

``` python
import nltk
nltk.download("words")
```

Note that `words` should be automatically installed by `pip` when you follow
the installation instructions for this package.

# Project layout

* `configs/`
  * `configs/experimetns` contains configuration files which specify hyperparamter settings for running experiments.
  * `configs/generation_data` contains configuration files for dataset generation
  * `configs/generation_prompt` contains configuration files for prompt generation based on the data previously stored
  * `configs/post_analysis` contains a configuration file which can be used for analysis of cost, latency, and performance metrics for one set of hyperparameters for a particular study. This config is used in the script scripts/anaylsis_for_one_study.py
  * `configs/post_hypothesis` contains a configuration file which specifies filtering criterias for generating figures for various hypotheses.
* `notebooks/` has a Jupyter notebook for generating figures that are used in the research paper
* `scripts/` contains Python scripts for running experiments, post-processing the results, and analysis of results
* `setlexsem/` is the module which has all the important functions and utils for analysis, experimentation, generation of data, samplers.
  * `analyze` contains code for error_analysis of post-processed results, visualizaiton code and utils neeeded for generating figures for hypothesis.
  * `experiment` contains code for invoking LLMs and running experiments for a particular hypothesis/study.
  * `generate` contains code for generating data, sample synthetic sets, prompts and utils needed for data generation.
  * `prepare` contains helper functions for partitioning words according to their frequencies.

# Generate the dataset

## Create the sets

### Sample sets of numbers or words

To make the CSV file containing sets of words and numbers, run:

```bash
python setlexsem/generate/generate_sets.py --config-path "configs/generation_sets/numbers_and_words.yaml" --seed-value 292 --save-data
```

### Sample sets based on training-set frequency

To sample sets based on their training-set frequency, we use an approximation based on rank frequency in the Google Books Ngrams corpus.

This requires `wget` (`brew install wget` or `apt install wget`). After
installing `wget`, you need to create `deciles.json`. The following command
downloads the English unigram term frequencies of the Google Books Ngram
corpus, filters them by the vocabulary of the nltk.words English vocabulary,
and stores the vocabulary, separated by deciles of rank frequency, in
`data/deciles.json`.

```bash
scripts/make-deciles.sh
```

This will take ~10 minutes or more, depending on your bandwidth and the speed of your computer.

To make the CSV file containing sets of words sampled by the approximated
training-set frequency, run:

```bash
python setlexsem/generate/generate_sets.py --config-path "configs/generation_sets/deciles.yaml" --seed-value 292 --save-data
```

### Sample "deceptive" sets

To sample semantically "deceptive" sets (see paper for details), create `hyponyms.json` by running the following command:

```bash
python scripts/make_hyponyms.py --output-path data/hyponyms.json
```

To make the CSV file containing deceptive sets:
```bash
python setlexsem/generate/generate_sets.py --config-path "configs/generation_sets/deceptive.yaml" --seed-value 292 --save-data
```

## Create the prompts

Once you've sampled the sets, create the prompts. The prompts are written as CSV files in the `prompts` directory.

### Example: Sets with numbers

To make the CSV file containing prompts sets of words and numbers, run:

```bash
python setlexsem/generate/generate_prompts.py --config_path "configs/generation_prompt/test_config.yaml"
```

## Run the evalution

1. Create a config file like `configs/experiments/anthr_sonnet.yaml`
2. Run the prompts:

  ```bash
  python setlexsem/experiment/run_experiments.py --account-number ACCOUNT_NUMBER --save-file --load-previous-run --config-file configs/experiments/anthr_sonnet.yaml
  ```

  **Note:** Currently, our experiments are dependent on AWS Bedrock and need an AWS account number to be provided. However, you have the capability to run experiments using OPENAI_KEY. We will add more instructions soon.

3. Post-process the results. (Check whether your `study_name` is present in the `STUDY2MODEL` dict in `setlexsem/constants.py`)

```bash
python scripts/save_processed_results_for_study_list.py
```

4, Analysis of cost, latency, and performance metrics for one set of hyperparameters for a particular study - enter hyperparameter values in the configs/post_analysis/study_config.json

```bash
python scripts/analysis_for_one_study.py
```

* Generate figures using notebooks/Hypothesis Testing - Manuscript.ipynb. Validate the filtering criterias in configs/post_hypothesis/hypothesis.json

## Test

To test the full-suite of tests, you need to provide the Account Number.

```bash
pytest -s .
```

You will be prompted to provide your Account Number after that.

### Coverage report

```bash
pip install pytest-cov
pytest --cov=setlexsem --cov-report=term-missing
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
