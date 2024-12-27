import ast
import logging
import os
import re
from typing import List, Literal

import pandas as pd
import yaml

from setlexsem.constants import (
    HPS,
    PATH_ANALYSIS,
    PATH_CONFIG_ROOT,
    PATH_POSTPROCESS,
    PATH_RESULTS_ROOT,
    STUDY2MODEL,
)
from setlexsem.generate.sample import make_sampler_name_from_hps

DEMONSTRATION_TYPES = [
    "formal_language",
    "plain_language",
    "functional_language",
    "pythonic_language",
    "iterative_accumulation",
]

# define the logger
logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.INFO)


def get_study_paths(
    sampler_hp, prompt_hp, random_seed, study_name, path_root
):
    """Generate study paths based on given parameters.

    Args:
        sampler_hp (tuple): Hyperparameters for the sampler.
        prompt_hp (tuple): Hyperparameters for the prompt.
        random_seed (int): Random seed for reproducibility.
        study_name (str): Name of the study.
        path_root (str): Root path for storing study results.
    """
    assert type(sampler_hp) is dict, "sampler_hp has to be a tuple"
    assert type(prompt_hp) is dict, "prompt_hp has to be a tuple"

    # get experiment_folder_structure
    experiment_folder_structure, filename_experiment = get_prompt_file_path(
        sampler_hp, prompt_hp, random_seed
    )

    # save df_results
    path_study_folder = os.path.join(
        path_root, study_name, experiment_folder_structure
    )

    # create path
    path_results = os.path.join(path_study_folder, filename_experiment)

    return path_study_folder, path_results


def get_prompt_file_path(sampler_hp, prompt_hp, random_seed):
    """
    Generate the experiment folder structure and filename based on given parameters.

    Args:
        sampler_hp (tuple): Hyperparameters for the sampler.
        prompt_hp (tuple): Hyperparameters for the prompt.
        random_seed (int): Random seed for reproducibility (dataset is generated with this seed)
    """
    sampler_name = make_sampler_name_from_hps(sampler_hp)
    filename_experiment = create_filename(
        prompt_hp["prompt_type"],
        sampler_name,
        prompt_hp["k_shot"],
        random_seed,
    )
    experiment_folder_structure = os.path.join(
        sampler_hp["set_types"],
        prompt_hp["op_list"],
        prompt_hp["prompt_approach"],
    )
    return experiment_folder_structure, filename_experiment


def get_data_filename(sampler_name, random_seed_value, num_runs):
    """Get the filename for the generated data"""

    return f"{sampler_name}_S-{random_seed_value}_Runs-{num_runs}.csv"


def create_param_format(sampler_name, random_seed_value):
    """Create a string for the parameters of the generated data"""
    return f"{sampler_name}_S-{random_seed_value}"


def create_filename(prompt_type, sampler_name, k_shot, random_seed_value):
    """Create the filename to save tables"""
    parameters_format = create_param_format(sampler_name, random_seed_value)
    return f"{prompt_type}_K-{k_shot}_{parameters_format}.csv"


def extract_values(filename):
    """Get the experiment values"""
    # If you change below, make sure you change the "paremeters_format" above
    # too.
    if "O-" in filename:
        matches = re.search(
            r"K-(\w+)_N-(\w+)_M-(\d+)_L-(\w+)_O-(\d+\.\d+)_S", filename
        )
        return (
            matches.group(1),
            matches.group(2),
            matches.group(3),
            matches.group(4),
            matches.group(5),
            "None",
        )
    elif "Decile-" in filename:
        matches = re.search(
            r"K-(\w+)_N-(\w+)_M-(\d+)_L-(\w+)_Decile-(\w+)_S", filename
        )
        return (
            matches.group(1),
            matches.group(2),
            matches.group(3),
            matches.group(4),
            "None",
            matches.group(5),
        )
    else:
        matches = re.search(r"K-(\w+)_N-(\w+)_M-(\d+)_L-(\w+)_S", filename)
        return (
            matches.group(1),
            matches.group(2),
            matches.group(3),
            matches.group(4),
            "None",
            "None",
        )


def make_object_set(df_in, column_name):
    """Convert the column name to a set (literal eval)"""
    df_temp = df_in[column_name].replace({-1: "{-1}"}).copy()
    try:
        object_set = df_temp.apply(ast.literal_eval)
    except ValueError:
        raise ValueError(f"Error converting {column_name} to set")
    return object_set


def create_results_df_from_folder(path_study):
    """
    Walk through directory and concatenate results with experiment parameters.
    """
    df_all = pd.DataFrame()
    for root, _, files in os.walk(path_study):
        for filename in files:
            if filename.endswith(".csv"):
                path_file = os.path.join(root, filename)
                df = pd.read_csv(path_file, dtype={"llm_vs_gt": bool})

                # add operation type and filename as new columns
                df["object_type"] = root.split("/")[-3]
                df["operation_type"] = root.split("/")[-2]
                df["prompt_approach"] = root.split("/")[-1]
                # remove extra info ()
                assert all(
                    df["op_name"] == df["operation_type"]
                ), "Error saving data: Operation name is inconsistent."
                df = df.drop(columns=["op_name"])

                k, n, m, item_len, overlap, deciles = extract_values(filename)
                if any(type in filename for type in DEMONSTRATION_TYPES):
                    df["prompt_type"] = [
                        type
                        for type in DEMONSTRATION_TYPES
                        if type in filename
                    ][0]
                else:
                    raise ValueError("No prompt type!")

                df["n_items"] = m
                df["k_shots"] = 0 if k == "None" else k
                df["overlap"] = overlap
                df["item_len"] = item_len
                if list(df["item_len"].unique())[0] is None:
                    df["max_value"] = n
                else:
                    df["max_value"] = -1

                set_columns = ["ground_truth", "result_obj", "set_A", "set_B"]
                for set_column in set_columns:
                    df[set_column] = make_object_set(
                        df, column_name=set_column
                    )

                # adjust type fo columns
                df["k_shots"] = df["k_shots"].astype(int)
                df["n_items"] = df["n_items"].astype(int)
                try:
                    df["item_len"] = df["item_len"].astype(int)
                except ValueError:
                    df["item_len"] = "None"
                try:
                    df["max_value"] = df["max_value"].astype(int)
                except ValueError:
                    df["max_value"] = "None"

                # add more info on deceptive status or decile status
                obj_type_list = list(df["object_type"].unique())
                assert (
                    len(obj_type_list) == 1
                ), "Error: more than one object type"
                df["is_deceptive"] = int(0)
                df["decile_num"] = int(-1)
                if deciles != "None":
                    df["decile_num"] = int(deciles)

                df["swapped"] = int(-1)
                if "deceptive" in obj_type_list[0].lower():
                    df["is_deceptive"] = int(1)
                    if "noswap" in root.lower():
                        df["swapped"] = int(0)
                    else:
                        df["swapped"] = int(1)

                df_all = pd.concat([df_all, df.reset_index(drop=True)])

    df_all = df_all.reset_index(drop=True)

    return df_all


def read_config(
    config_path=os.path.join(PATH_CONFIG_ROOT, "config.yaml")
) -> dict:
    """
    Read the configuration file and return a dictionary of the configuration
    """
    with open(config_path, "r") as f:
        LOGGER.info(f"loading {config_path}")
        config = yaml.safe_load(f)

    # Validate and assign the configuration values
    STUDY_NAME: str = (
        config["STUDY_NAME"] if "STUDY_NAME" in config else "None"
    )
    N_RUN: int = config["N_RUN"]
    LOAD_GENERATED_DATA: bool = (
        config["LOAD_GENERATED_DATA"]
        if "LOAD_GENERATED_DATA" in config
        else False
    )
    RANDOM_SEED_VAL: int = config["RANDOM_SEED_VAL"]
    OP_LIST: List[str] = config["OP_LIST"]
    MODEL_NAME: str = (
        config["MODEL_NAME"] if "MODEL_NAME" in config else "None"
    )

    SET_TYPES: List[str] = config["SET_TYPES"]
    N: List[int] = config["N"]
    if "M" in config.keys():  # backward compatibility
        M_A: List[int] = config["M"]
        M_B: List[int] = config["M"]
    else:
        M_A: List[int] = config["M_A"]
        M_B: List[int] = config["M_B"]
    ITEM_LEN: List[int] = config["ITEM_LEN"]
    OVERLAP_FRACTION: List[int] = config["OVERLAP_FRACTION"]
    DECILE_NUM: List[int] = config["DECILE_NUM"]

    K_SHOT: List[int] = config["K_SHOT"]
    PROMPT_TYPE: List[Literal["formal_language"]] = config["PROMPT_TYPE"]
    PROMPT_APPROACH: List[Literal["baseline"]] = config["PROMPT_APPROACH"]
    IS_FIX_SHOT: List[bool] = config["IS_FIX_SHOT"]

    return {
        "STUDY_NAME": STUDY_NAME,
        "N_RUN": N_RUN,
        "LOAD_GENERATED_DATA": LOAD_GENERATED_DATA,
        "RANDOM_SEED_VAL": RANDOM_SEED_VAL,
        "OP_LIST": OP_LIST,
        "MODEL_NAME": MODEL_NAME,
        "SET_TYPES": SET_TYPES,
        "N": N,
        "M_A": M_A,
        "M_B": M_B,
        "ITEM_LEN": ITEM_LEN,
        "OVERLAP_FRACTION": OVERLAP_FRACTION,
        "DECILE_NUM": DECILE_NUM,
        "K_SHOT": K_SHOT,
        "PROMPT_TYPE": PROMPT_TYPE,
        "PROMPT_APPROACH": PROMPT_APPROACH,
        "IS_FIX_SHOT": IS_FIX_SHOT,
    }


# postprocess lm results
def fix_response_format(x):
    """Replace empty set with empty dict"""
    try:
        if x in [{"theemptyset"}, {"emptyset"}, {"Theemptyset"}]:
            x = set()
        return x
    except Exception as e:
        LOGGER.warning("response is not a string: {e}")
        return set()


# Postprocessing Results
def aggregate_metrics(x):
    """Calculate accuracy and number of samples"""
    avg_accuracy = round(x["accuracy"].mean() * 100, 2)
    avg_precision = round(x["precision"].mean() * 100, 2)
    avg_recall = round(x["recall"].mean() * 100, 2)
    avg_jaccard_index = round(x["jaccard_index"].mean() * 100, 2)
    avg_percent_match = round(x["percent_match"].mean(), 2)
    count = x["accuracy"].count()
    return pd.Series(
        [
            avg_accuracy,
            avg_precision,
            avg_recall,
            avg_jaccard_index,
            avg_percent_match,
            count,
        ],
        index=[
            "avg_accuracy",
            "avg_precision",
            "avg_recall",
            "avg_jaccard_index",
            "avg_percent_match",
            "n_samples",
        ],
    )


def get_accuracy_metrics(ground_truth, model_output):
    """Get the accuracy metrics from comparing ground-truth and model-output"""
    # Ensure dataset is a set
    try:
        if not isinstance(ground_truth, set):
            ground_truth = ast.literal_eval(ground_truth)
        if not isinstance(model_output, set):
            model_output = ast.literal_eval(model_output)
    except ValueError:
        raise ValueError("Error converting to set")

    # Calculate intersection and union
    intersection = ground_truth.intersection(model_output)
    union = ground_truth.union(model_output)

    # Calculate metrics
    accuracy = ground_truth == model_output
    precision = (
        len(intersection) / len(model_output) if len(model_output) > 0 else 0
    )
    recall = (
        len(intersection) / len(ground_truth) if len(ground_truth) > 0 else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0
    exact_match = (
        1 if ground_truth == model_output else 0
    )  # this is the same as accuracy
    percent_match = (
        len(intersection) / len(ground_truth) * 100
        if len(ground_truth) > 0
        else 0
    )

    return pd.Series(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "jaccard_index": jaccard_index,
            "exact_match": exact_match,
            "percent_match": percent_match,
        }
    )


def save_processed_results(study_name, hps=HPS, overwrite=False):
    """Save processed results of a study"""
    model_name = STUDY2MODEL[study_name]
    path_study = os.path.join(PATH_RESULTS_ROOT, study_name)
    path_study_all_runs_raw = os.path.join(
        PATH_POSTPROCESS, f"{study_name}_all_runs.csv"
    )
    path_aggregated_results = os.path.join(
        PATH_POSTPROCESS, f"{study_name}.csv"
    )
    os.makedirs(PATH_ANALYSIS, exist_ok=True)
    path_analysis = os.path.join(PATH_ANALYSIS, f"{study_name}.csv")

    # data regarding all runs
    if os.path.exists(path_study_all_runs_raw) and not overwrite:
        LOGGER.info(f"Raw file is already processed, loading: {study_name}")
        df_all_runs = pd.read_csv(path_study_all_runs_raw)
    else:
        df_all_runs = create_results_df_from_folder(path_study)
        df_all_runs.to_csv(path_study_all_runs_raw, index=False)

    if df_all_runs.empty:
        LOGGER.warning(f"No data found for {study_name}")
        return pd.DataFrame(), pd.DataFrame()

    if os.path.exists(path_aggregated_results) and not overwrite:
        LOGGER.info(f"File is already processed, loading: {study_name}")
        df_all_runs = pd.read_csv(path_analysis)
        df_results = pd.read_csv(path_aggregated_results)
        return df_all_runs, df_results

    # fix response formatting issues
    df_all_runs["result_obj"] = df_all_runs["result_obj"].apply(
        fix_response_format
    )

    # postprocess results
    LOGGER.info(f"Postprocessing results for {study_name}")
    df_analysis = df_all_runs.apply(
        lambda x: get_accuracy_metrics(x["ground_truth"], x["result_obj"]),
        axis=1,
    )
    df_all_runs = pd.concat([df_all_runs, df_analysis], axis=1)

    # aggregating all runs and calculate accuracy for each hyperparameter set
    metric_list = [
        "accuracy",
        "precision",
        "recall",
        "jaccard_index",
        "percent_match",
    ]
    # fix grouping when item-length is NaN (unconstrained)
    df_all_runs["item_len"] = (
        df_all_runs["item_len"].replace("None", pd.NA).fillna(-1).astype(int)
    )
    df_results = (
        df_all_runs.groupby(hps)[metric_list]
        .apply(aggregate_metrics)
        .reset_index()
    )
    df_results["n_samples"] = df_results["n_samples"].astype(int)
    df_results["model_name"] = model_name
    df_results["is_deceptive"] = df_results["is_deceptive"].astype(int)
    df_results["decile_num"] = df_results["decile_num"].astype(int)
    df_results["study_name"] = study_name

    # save aggregation
    df_results.to_csv(path_aggregated_results, index=False)
    # save analysis
    df_all_runs.to_csv(path_analysis, index=False)
    LOGGER.info(f"Saved processed results for {study_name}")

    return df_all_runs, df_results


def load_processed_data(csv_path):
    """Load the processed data"""
    df_all = pd.read_csv(csv_path)
    df = df_all.iloc[:, :]
    df = assign_types(df)
    return df


def assign_types(df):
    """Assign data types"""
    # convert prompt approach to category ordered
    df["prompt_approach"] = df["prompt_approach"].astype("category")
    # convert object_type to category ordered with order of "numbers", "words"
    df["object_type"] = df["object_type"].replace(
        {
            "overlapping_BasicNumberSampler": "overlap numbers",
            "overlapping_BasicWordSampler": "overlap words",
        }
    )
    df["object_type"] = df["object_type"].astype("category")
    # categorize prompt type (formal_language, plain_language)
    df["prompt_type"] = df["prompt_type"].astype("category")
    # categorize prompt type (formal_language, plain_language)
    df["item_len"] = df["item_len"].replace({"None": -1})
    df["item_len"] = df["item_len"].fillna(-1)
    df["item_len"] = df["item_len"].astype(int)

    df["is_deceptive"] = df["is_deceptive"].astype(int)
    df["decile_num"] = df["decile_num"].astype(int)

    ordered_operations = [
        "union",
        "intersection",
        "difference",
        "symmetric difference",
        # "cartesian product",
    ]

    df["operation_type"] = df["operation_type"].astype("category")
    df["operation_type"] = df["operation_type"].cat.set_categories(
        ordered_operations, ordered=True
    )

    phrasing_list = [
        "formal_language",
        "plain_language",
        "functional_language",
        "pythonic_language",
        "iterative_accumulation",
    ]
    df["prompt_type"] = df["prompt_type"].astype("category")
    df["prompt_type"] = df["prompt_type"].cat.set_categories(
        phrasing_list, ordered=True
    )

    model_list = [
        "instant",
        "haiku",
        "sonnet",
        "gpt35",
        "mistralL",
        "mistralS",
        "llama",
    ]
    df["model_name"] = df["model_name"].astype("category")
    df["model_name"] = df["model_name"].cat.set_categories(
        model_list, ordered=True
    )
    return df


def convert_model_name(model_name):
    return {
        "instant": "Claude Instant",
        "sonnet": "Claude Sonnet",
        "haiku": "Claude Haiku",
        "gpt35": "GPT 3.5",
        "mistralL": "Mistral Large",
    }[model_name]


def make_nice(df_in):
    """Converts the code names to camera-ready names"""
    nice_map = {
        "instant": "Claude Instant",
        "sonnet": "Claude Sonnet",
        "haiku": "Claude Haiku",
        "gpt35": "GPT 3.5-Turbo-0613",
        "mistralL": "Mistral Large-2402",
        "mistralS": "Mistral Small-2402",
        "llama": "Meta Llama3-70b",
        "formal_language": "Formal Language",
        "plain_language": "Plain Language",
        "baseline": "Base Prompt",
        "baseline_allow_empty": "Base Prompt\n(allow empty set)",
        "composite": "CoT and Self-Reflection",
        "composite_allow_empty": "CoT and Self-Reflection\n(allow empty set)",
        "decile_num": "Frequency decile",
        "overlap numbers": "Overlapping Sets - Numbers",
        "overlap words": "Overlapping Sets - Words",
        "numbers": "Numbers",
        "words": "Words",
        "decile_words": "Words by Frequency",
        "deceptive_words": "Deceptive Words",
        "union": "Union",
        "intersection": "Intersection",
        "symmetric difference": "Symmetric\ndifference",
        "difference": "Difference",
        "is_deceptive": "Token similarity",
        "n_items": "Operand size",
        "operation_type": "Set operation",
        "object_type": "Token type",
        "item_len": "Token length",
        "prompt_approach": "Prompting method",
        "model_name": "LLM",
        "prompt_type": "Demonstration phrasing",
        "decile_num": "Token frequency",
        "k_shots": "Number of demonstrations",
        "swapped": "Relationship between sets A and B",
        "functional_language": "Functional Language",
        "pythonic_language": "Pythonic Language",
        "iterative_accumulation": "Iterative Accumulation",
    }

    ugly_map = {v: k for k, v in nice_map.items()}
    return (
        df_in.rename(columns=nice_map)
        .replace(nice_map)
        .rename(
            columns={
                col: " ".join([word.capitalize() for word in col.split("_")])
                for col in df_in.columns
            }
        )
    ), ugly_map
