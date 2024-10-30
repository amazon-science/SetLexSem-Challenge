import os

import pandas as pd

from setlexsem.constants import PATH_ROOT


def create_fig_path(hypo_name, folder="appendix"):
    figure_path = os.path.join(
        PATH_ROOT, f"manuscript/{folder}/{hypo_name}.pdf"
    )
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    return figure_path


def agg(x):
    """Aggregation to calculate min, avg, max"""
    min_val = x.min()
    max_val = x.max()
    mean_val = x.mean()
    std_val = x.std()
    return pd.Series(
        [mean_val, std_val, min_val, max_val],
        index=["Avg", "Std", "Min", "Max"],
    )


def get_stats(df_in):
    """Create a tabular data to compare groups based on min, avg, max"""
    return (
        # df_in.groupby(["operation_type", "object_type"], observed=True)["accuracy"]
        df_in.groupby(["Set operation", "Token type"], observed=True)[
            "Accuracy"
        ]
        .apply(agg)
        .unstack()
        .transpose()
        .round(2)
    )


def get_config(dataframe):
    text_out = ""
    for col in [
        "Token type",
        "Set operation",
        "Demonstration phrasing",
        "Prompting method",
        "Operand size",
        "Number of demonstrations",
        "Token length",
        "Token similarity",
        "Token frequency",
        "Relationship between sets A and B",
        "N Samples",
        "LLM",
    ]:
        # get unique values and report as text
        try:
            text_out += f"{col:35s}: {dataframe[col].unique().tolist()}\n"
        except:
            pass

    return text_out


def save_config_and_data(dataframe, hypo_name, supp_root):
    file_config_csv = os.path.join(supp_root, f"{hypo_name}.csv")
    dataframe.to_csv(file_config_csv, index=False)

    # save config_text file in file_config_txt
    file_config_txt = os.path.join(supp_root, f"{hypo_name}_config.txt")
    config_text = get_config(dataframe)
    with open(file_config_txt, "w") as f:
        f.write(config_text)


def create_filtered_df_for_hypothesis(df, hypothesis_config):
    # Construct filter condition dynamically
    condition = pd.Series([True] * len(df))

    for key, values in hypothesis_config.items():
        if key != "hypo_name":
            # Check if column values are in the list (this handles OR inside each key)
            condition &= df[key].isin(values)

    # Apply the filter
    df_new = df[condition]

    return df_new


def add_nl(x):
    # Add new line
    return str(x) + "\n"


def add_text(x):
    # Map Deciles
    if x == 1:
        return f"{x}st decile"
    elif x == 2:
        return f"{x}nd decile"
    elif x == 3:
        return f"{x}rd decile"
    else:
        return f"{x}th decile"


# Function to convert set strings to sets and concatenate them
def concat_sets(row):
    set1 = eval(row["A"])
    set2 = eval(row["B"])
    return set1.union(set2)
