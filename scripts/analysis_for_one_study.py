import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd

from setlexsem.analyze.error_analysis import (
    create_error_analysis_table,
    filter_dataframe,
)
from setlexsem.constants import (
    HPS,
    PATH_ANALYSIS,
    PATH_ANALYSIS_CONFIG_ROOT,
    PATH_CONFIG_ROOT,
)
from setlexsem.experiment.lmapi import PRICING_PER_TOKEN
from setlexsem.utils import read_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-filename",
        default="study_config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--grouping-items",
        nargs="+",
        default=["object_type", "operation_type"],
        help="Items to group by",
    )
    args = parser.parse_args()
    return args


def validate_analysis_config_file(study_dict):
    """Validation function to ensure the configuration is correct"""
    assert "study_name" in study_dict
    assert "object" in study_dict
    assert "operation" in study_dict
    assert "item_len" in study_dict
    if "n_items" not in study_dict:
        assert "n_items_in_A" in study_dict
        assert "n_items_in_B" in study_dict
    assert "k_shots" in study_dict
    assert "prompt_approach" in study_dict


def main():
    args = parse_args()
    config_filename = args.config_filename
    grouping_items = args.grouping_items

    with open(os.path.join(PATH_ANALYSIS_CONFIG_ROOT, config_filename)) as f:
        study_config = json.load(f)
        validate_analysis_config_file(study_config)

    study_name = study_config["study_name"]
    print(f"\nStudy Name: {study_name}")
    df_study = pd.read_csv(os.path.join(PATH_ANALYSIS, f"{study_name}.csv"))

    print("-" * 50)
    print("Stats for the study: ")
    print(df_study.groupby(grouping_items)["llm_vs_gt"].mean() * 100)
    print("-" * 25)

    model_name = read_yaml(
        os.path.join(PATH_CONFIG_ROOT, "study_to_models.yaml")
    )[study_name]

    try:
        price_in = PRICING_PER_TOKEN[model_name]["price_in"]
        price_out = PRICING_PER_TOKEN[model_name]["price_out"]
    except KeyError:
        raise KeyError("Invalid study name for cost calculation")

    token_in = df_study["context_length_in"].sum()
    token_out = df_study["context_length_out"].sum()

    print(
        f"""Cost Analysis
    In:  {token_in:<7,} (${price_in*token_in:,.5f})
    Out: {token_out:<7,} (${price_out*token_out:,.5f})
    """
    )
    print("-" * 25)
    print("Token Comparison:")
    print(
        df_study.groupby(HPS)[
            ["context_length_in", "context_length_out"]
        ].mean()
    )

    print("-" * 25)
    filter_dict = {
        "object_type": study_config["object"],
        "operation_type": study_config["operation"],
        "item_len": study_config["item_len"],
        "prompt_approach": study_config["prompt_approach"],
        "k_shots": study_config["k_shots"],
        "swapped": study_config["swapped"],
        "max_value": -1,
    }
    # add number of members
    if "n_items" in study_config:
        filter_dict["n_items"] = study_config["n_items"]
    elif "n_items_in_A" in study_config:
        filter_dict["n_items_in_A"] = study_config["n_items_in_A"]
        filter_dict["n_items_in_B"] = study_config["n_items_in_B"]
    else:
        raise ValueError("Invalid study configuration")
    df_exp = filter_dataframe(df_study, filter_dict)

    if not df_exp.empty:
        df_temp = create_error_analysis_table(df_exp, index_dict=filter_dict)
        print("Detailed stats for the hyperparameters combination:\n")
        print(df_temp.T)
    else:
        print(
            f"""Warning: No data found for the following hyperparameters combination:
{"\n".join(f"- {k}: {v}" for k, v in study_config.items())}"""
        )

    print("-" * 50)


if __name__ == "__main__":
    main()
