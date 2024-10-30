# Overview
This research repository maintains the codes and the results for the research paper: SETLEXSEM CHALLENGE: Using Set Operations to Evaluate the Lexical and Semantic Robustness of Language Models

"Set theory has become the standard foundation for mathematics, as every mathematical object can be viewed as a set." - https://plato.stanford.edu/entries/set-theory/

# Install

When installing, it's important to upgrade to the most recent pip. This ensures that `setup.py` runs correctly. An outdated version of pip can fail to run the `InstallNltkWordnetAfterPackages` command class in setup.py and cause subsequent errors.

``` bash
/usr/bin/python3 -mvenv venv
. venv/bin/activate
python3 -m pip install --upgrade pip
pip install -e .
pip install -e ."[dev, test]"
```

## NLTK words

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
* `notebooks/ ` has a Jupyter notebook for generating figures that are used in the research paper
* `scripts/` contains Python scripts for running experiments, post-processing the results, and analysis of results
* `setlexsem/` is the module which has all the important functions and utils for analysis, experimentation, generation of data, samplers.
    * `analyze` contains code for error_analysis of post-processed results, visualizaiton code and utils neeeded for generating figures for hypothesis.
    * `experiment` contains code for invoking LLMs and running experiments for a particular hypothesis/study.
    * `generate` contains code for generating data, sample synthetic sets, prompts and utils needed for data generation.
    * `prepare` contains helper functions for partitioning words according to their frequencies.

## End-to-End

* Check the configs/anthr_sonnet.yaml
* Running the experiment
```
python mathbedrock/experiment/run_experiments.py --account_number <account-number> --save_file 1 --load_last_run 1 --config_file configs/anthr_sonnet.yaml
```
* Post-Processing results (Check whether your study_name is present in the STUDY2MODEL dict in setlexsem/constants.py)
```
python scripts/save_processed_results_for_study_list.py
```
* Analysis of cost, latency, and performance metrics for one set of hyperparameters for a particular study - enter hyperparameter values in the analysis_config/study_config.json
```
python scripts/analysis_for_one_study.py
```
* Generate figures using notebooks/Hypothesis Testing - Manuscript.ipynb. Validate the filtering criterias in hypothesis_config/hypothesis.json


# Test
To test the full-suite of tests, you need to provide the Account Number.
```
pytest -s .
```
You will be prompted to provide your Account Number after that.

# Coverage Report
```
pip install pytest-cov
pytest --cov=setlexsem --cov-report=term-missing
```

# TO-DO
* Add instructions for how to create deciles.json (see the code in scripts/).
  * Requires `wget`
    * Mac OS: `brew install wget`
  * Clean up code in scripts/. Nick ran this code once after submitsion to AMLC and to NeurIPS and found that most or all of the storage.googleapis.com URLs now return HTTP 404. Even the links on https://storage.googleapis.com/books/ngrams/books/datasetsv3.html did not work. Should we document an altogether different way to obtain the ngram frequencies?

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

