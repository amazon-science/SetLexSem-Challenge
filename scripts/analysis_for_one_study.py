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


def validate_analysis_config_file(study_dict):
    assert "study_name" in study_dict
    assert "object" in study_dict
    assert "operation" in study_dict
    assert "item_len" in study_dict
    assert "n_items" in study_dict
    assert "k_shots" in study_dict
    assert "prompt_approach" in study_dict


def main():
    with open(
        os.path.join(PATH_ANALYSIS_CONFIG_ROOT, "study_config.json"), "r"
    ) as f:
        study_config = json.load(f)
        validate_analysis_config_file(study_config)
        study_conf = SimpleNamespace(**study_config)

    study_name = study_conf.study_name

    print(f"\nStudy Name: {study_name}")

    df_study = pd.read_csv(os.path.join(PATH_ANALYSIS, f"{study_name}.csv"))

    print("\nStats for the study: ")
    print(
        df_study.groupby(["decile_num", "object_type", "operation_type"])[
            "llm_vs_gt"
        ].mean()
        * 100
    )

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
        f"""\nCost Analysis
    In:  {token_in:<7,} (${price_in*token_in:,.5f})
    Out: {token_out:<7,} (${price_out*token_out:,.5f})
    """
    )

    print("\nToken Comparison:")
    print(
        df_study.groupby(HPS)[
            ["context_length_in", "context_length_out"]
        ].mean()
    )

    filter_dict = {
        "object_type": study_conf.object,
        "operation_type": study_conf.operation,
        "item_len": study_conf.item_len,
        "prompt_approach": study_conf.prompt_approach,
        "n_items": study_conf.n_items,
        "k_shots": study_conf.k_shots,
        "max_value": -1,
    }

    df_exp = filter_dataframe(df_study, filter_dict)

    if not df_exp.empty:
        df_temp = create_error_analysis_table(df_exp, index_dict=filter_dict)

        print("\n\n Detailed stats for the hyperparameters combination:\n")
        print(df_temp)
    else:
        print(
            f"""\n\nNo data found for the following hyperparameters combination:
            object : {study_conf.object}
            operation : {study_conf.operation}
            item_len : {study_conf.item_len}
            prompt_approach: {study_conf.prompt_approach}
            n_items: {study_conf.n_items}
            k_shots: {study_conf.k_shots}"""
        )


if __name__ == "__main__":
    main()
