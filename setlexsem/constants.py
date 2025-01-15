import os
from pathlib import Path


def get_path_root(this_file):
    path = Path(this_file)
    if not "setlexsem" in path.parts:
        raise ValueError("Can't find root of repo. 'setlexsem' not in path.")
    last_part = path.parts.index("setlexsem")
    path_root = os.path.sep + os.path.sep.join(path.parts[1:last_part])
    return path_root


# get root-directory
PATH_ROOT = get_path_root(__file__)

PATH_DATA_ROOT = os.path.join(PATH_ROOT, "data")
PATH_PROMPTS_ROOT = os.path.join(PATH_ROOT, "prompts")
PATH_RESULTS_ROOT = os.path.join(PATH_ROOT, "results")
PATH_CONFIG_ROOT = os.path.join(PATH_ROOT, "configs")
PATH_ANALYSIS_CONFIG_ROOT = os.path.join(PATH_CONFIG_ROOT, "post_analysis")
PATH_HYPOTHESIS_CONFIG_ROOT = os.path.join(
    PATH_CONFIG_ROOT, "post_hypothesis"
)
PATH_POSTPROCESS = os.path.join(PATH_ROOT, "processed_results")
PATH_ANALYSIS = os.path.join(PATH_ROOT, "analysis")

ACCOUNT_NUMBER = None
if os.path.exists(os.path.join(PATH_ROOT, "secrets.txt")):
    with open(os.path.join(PATH_ROOT, "secrets.txt"), "r") as file:
        ACCOUNT_NUMBER = int(file.read())

HPS = [
    "object_type",
    "operation_type",
    "prompt_type",
    "prompt_approach",
    "n_items",
    "k_shots",
    "item_len",
    "max_value",
    "is_deceptive",
    "decile_num",
    "swapped",
]

TOKEN_ORDER = [
    "1st decile",
    "2nd decile",
    "3rd decile",
    "4th decile",
    "5th decile",
    "6th decile",
    "7th decile",
    "8th decile",
    "9th decile",
]

STUDY2DECEPTIVE_WORD_SAMPLER = {
    "240531_ClaudeHaiku": "Random Baseline",
    "240508_ClaudeHaiku_Deceptive": "Deceptive, swapped",
    "240508_ClaudeHaiku_Deceptive_NoSwap": "Deceptive, not swapped",
}
