# Overview

This research repository maintains the code and the results for the research paper: SETLEXSEM CHALLENGE: Using Set Operations to Evaluate the Lexical and Semantic Robustness of Language Models.

_"Set theory has become the standard foundation for mathematics, as every mathematical object can be viewed as a set."_ -[Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/entries/set-theory/)

## Install

When installing, it's important to upgrade to the most recent pip. This ensures that `setup.py` runs correctly. An outdated version of pip can fail to run the `InstallNltkWordnetAfterPackages` command class in setup.py and cause subsequent errors.

``` bash
/usr/bin/python3 -mvenv venv
. venv/bin/activate
python3 -m pip install --upgrade pip
pip install -e .
pip install -e ."[dev, test]"
```

### NLTK words

If you get errors from `nltk` about package `words` not being installed while
executing the code in this repository, run:

``` python
import nltk
nltk.download("words")
```

Note that `words` should be automatically installed by `pip` when you follow
the installation instructions for this package.

## Directory Structure

* `analysis_config/` contains a configuration file which can be used for analysis of cost, latency, and performance metrics for one set of hyperparameters for a particular study. This config is used in the script scripts/anaylsis_for_one_study.py
* `config/` contains configuration files which specify hyperparamter settings for running experiments.
* `hypothesis_config/` contains a configuration file which specifies filtering criterias for generating figures for various hypotheses.
* `notebooks/` has a Jupyter notebook for generating figures that are used in the research paper
* `scripts/` contains Python scripts for running experiments, post-processing the results, and analysis of results
* `setlexsem/` is the module which has all the important functions and utils for analysis, experimentation, generation of data, samplers.
  * `analyze` contains code for error_analysis of post-processed results, visualizaiton code and utils neeeded for generating figures for hypothesis.
  * `experiment` contains code for invoking LLMs and running experiments for a particular hypothesis/study.
  * `generate` contains code for generating data, sample synthetic sets, prompts and utils needed for data generation.
  * `prepare` contains helper functions for partitioning words according to their frequencies.

## Data Generation

### Numbers and Words

To generate your own data, you can run the following:

```bash
python setlexsem/generate/generate_data.py --config_path "configs/data_generation/numbers_and_words.yaml" --seed_value 292 --save_data 1
```

### Deciles

To generate data based on `deciles`, you need to have `wget` installed (`brew install wget`). After installing `wget`, you need to create the `deciles.json` file. This process may take ~5-10 minutes.

```bash
scripts/raw-data-google-ngram-run.sh
scripts/partition_words.py --args ...
python scripts/partition_words.py \
    --k 10 \
    --google-ngram-path PATH/googlebooks-eng-all-1gram-20120701.filtered \
    --output-path deciles.json
```

Then, you can run the following to generate data:

```bash
python setlexsem/generate/generate_data.py --config_path "configs/data_generation/deciles.yaml" --seed_value 292 --save_data 1
```

## Prompt Generation

### Numbers

To generate your own data, you can run the following:

```bash
python setlexsem/generate/generate_prompts.py --config_path "configs/prompt_generation/test_config.yaml" --seed_value 292 --save_data 1
```

## Running Experiments End-to-End

* Check the configs/anthr_sonnet.yaml
* Running the experiment

```bash
python setlexsem/experiment/run_experiments.py --account_number <account-number> --save_file 1 --load_last_run 1 --config_file configs/experiments/anthr_sonnet.yaml
```

* Post-Processing results (Check whether your study_name is present in the STUDY2MODEL dict in setlexsem/constants.py)

```bash
python scripts/save_processed_results_for_study_list.py
```

* Analysis of cost, latency, and performance metrics for one set of hyperparameters for a particular study - enter hyperparameter values in the analysis_config/study_config.json

```bash
python scripts/analysis_for_one_study.py
```

* Generate figures using notebooks/Hypothesis Testing - Manuscript.ipynb. Validate the filtering criterias in hypothesis_config/hypothesis.json

## Test

To test the full-suite of tests, you need to provide the Account Number.

```bash
pytest -s .
```

You will be prompted to provide your Account Number after that.

### Coverage Report

```bash
pip install pytest-cov
pytest --cov=setlexsem --cov-report=term-missing
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
