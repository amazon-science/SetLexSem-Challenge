from ast import literal_eval

import pandas as pd


def filter_dataframe(df_in, filter_criteria):
    """
    Filter a dataframe based on filter criteria

    Args:
        df (pd.DataFrame): Dataframe to filter
        filter_criteria (dict): Dictionary of column names and values to filter

    Returns:
        pd.DataFrame: Filtered dataframe
    """

    # Make a copy of the input dataframe
    filtered_df = df_in.copy()

    # Iteratively filter dataframe based on filter criteria
    for col, value in filter_criteria.items():
        if isinstance(value, list):
            filtered_df = filtered_df[filtered_df[col].isin(value)]
        else:
            filtered_df = filtered_df[filtered_df[col] == value]

    # Drop columns used for filtering except "llm_vs_gt"
    # drop_cols = [col for col in filter_criteria if col != "llm_vs_gt"]
    # filtered_df.drop(columns=drop_cols, inplace=True)

    # Reset index and return filtered dataframe
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df


def calculate_extra_info(df_in):
    """Calculate extra information we need for error-analysis row-by-row
    This is the main chunk of comparisons that we are doing row-by-row"""
    df_out = df_in.copy()

    df_out["did_not_follow_instruction"] = (
        (df_out["result_obj"] == -1)
        | (df_out["result_obj"].apply(lambda x: x == {-1}))
    ).astype(int)

    # replace -1 to a set
    df_out["result_obj"] = df_out["result_obj"].replace({-1: {-1}})

    # literal_eval lines are added to convert strings to sets
    # Without the eval, we run into operation '&, ^' not supported for str and str operands
    df_out["res_len_diff"] = df_out.apply(
        lambda x: len(
            literal_eval(x["ground_truth"]) ^ literal_eval(x["result_obj"])
        ),
        axis=1,
    )

    df_out["res_mismatch_with_A"] = df_out.apply(
        lambda x: len(
            (literal_eval(x["set_A"]) & literal_eval(x["ground_truth"]))
            ^ (literal_eval(x["set_A"]) & literal_eval(x["result_obj"]))
        ),
        axis=1,
    )

    df_out["res_mismatch_with_B"] = df_out.apply(
        lambda x: len(
            (literal_eval(x["set_B"]) & literal_eval(x["ground_truth"]))
            ^ (literal_eval(x["set_B"]) & literal_eval(x["result_obj"]))
        ),
        axis=1,
    )

    df_out["res_made_up_numbers"] = df_out.apply(
        lambda x: len(
            literal_eval(x["set_B"])
            | literal_eval(x["set_A"])
            | literal_eval(x["result_obj"])
        )
        - len(literal_eval(x["set_A"]) | literal_eval(x["set_B"])),
        axis=1,
    )

    df_out["res_len_comparison"] = df_out.apply(
        lambda x: f'G{len(literal_eval(x["ground_truth"]))} vs. L{len(literal_eval(x["result_obj"]))}',
        axis=1,
    )

    return df_out


def get_normalized_count(df_check, column_name="res_len_comparison"):
    """This is a helper to get the percentage of repetition of values"""
    return list(
        df_check[column_name]
        .value_counts(normalize=True)
        .apply(lambda x: 100 * x)  # calculate percentage
        .round(2)
        .to_dict()
        .items()
    )


def create_error_analysis_table(df_in, index_dict):
    """Aggregate all metrics and report the final table"""
    gt_colname = "ground_truth"
    lm_colname = "result_obj"

    # prepare dataframe
    df_in_copy = df_in.copy()
    df_out = calculate_extra_info(df_in_copy)
    df_out["res_len_comparison"] = df_out["res_len_comparison"].astype(
        "category"
    )
    n_data = len(df_out)

    # other metrics
    df_wrong = df_out.query("llm_vs_gt == False")
    list_wrong = get_normalized_count(df_wrong)

    df_correct = df_out.query("llm_vs_gt == True")
    list_correct = get_normalized_count(df_correct)

    # overall list
    empty_set_equal = (df_out[gt_colname] == set()) & (
        df_out[lm_colname] == set()
    )
    empty_set_mismatch = (df_out[gt_colname] == set()) & (
        df_out[lm_colname] != set()
    )

    df_out_nonempty = df_out[df_out[gt_colname] != set()]

    # prepare dict
    dict_analysis = {}
    # insert the index dictionary
    dict_analysis.update(index_dict)

    dict_analysis.update(
        {
            # percentages
            "accuracy": round((df_out.llm_vs_gt.sum() / n_data) * 100, 2),
            "accuracy_non_empty": round(
                (df_out_nonempty.llm_vs_gt.sum() / n_data) * 100, 2
            ),
            "pct_nullset_correct": round(
                (empty_set_equal.sum() / n_data) * 100, 2
            ),
            "pct_nullset_wrong": round(
                (empty_set_mismatch.sum() / n_data) * 100, 2
            ),
            "pct_with_made_up_vals": (
                df_out.res_made_up_numbers.sum() / n_data
            )
            * 100,
            # count metrics
            "n_comparisons": len(df_out),
            "n_correct": df_out.llm_vs_gt.sum(),
            "n_wrong": len(df_out) - df_out.llm_vs_gt.sum(),
            "empty_set_equal_count": empty_set_equal.sum(),
            "empty_set_mismatch_count": empty_set_mismatch.sum(),
            "made_up_vals_sum": df_out.res_made_up_numbers.sum(),
            "did_not_follow_instruction": (
                df_out.did_not_follow_instruction.sum()
            ),
            # compare metrics
            "top1_mistake": (
                str(list_wrong[0]) if len(list_wrong) >= 1 else None
            ),
            "top2_mistake": (
                str(list_wrong[1]) if len(list_wrong) >= 2 else None
            ),
            "top3_mistake": (
                str(list_wrong[2]) if len(list_wrong) >= 3 else None
            ),
            "top1_correct": (
                str(list_correct[0]) if len(list_correct) >= 1 else None
            ),
            "top2_correct": (
                str(list_correct[1]) if len(list_correct) >= 2 else None
            ),
            "top3_correct": (
                str(list_correct[2]) if len(list_correct) >= 3 else None
            ),
        }
    )

    # make dataframe so keys are column names
    df_analysis = pd.DataFrame(dict_analysis, index=[0])

    return df_analysis
