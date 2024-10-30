import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["legend.fontsize"] = 8


NAME_SET_OP = "Set operation"

METRIC_TO_VISUALIZE = "Avg Accuracy"


# Avg Accuracy	Avg Precision	Avg Recall	Avg Jaccard Index	Avg Percent Match
def viz_barplot(
    df_res,
    hue_group,
    break_by=None,
    filter_query=None,
    save_fig="",
    legend_loc="upper right",
    plot_type="bar",
    figure_size=(8, 3),
):
    """
    Create a bar plot to visualize the accuracy by operation type.

    Args:
        df_res (pandas.DataFrame): The input DataFrame containing the data.
        hue_group (str): The column name to group the data by
            e.g., 'model_name'
        break_by (str, optional): The column name to create subplots for.
            If None, a single plot is created.
        filter_query (str, optional): A query string to filter the DataFrame.

    Returns:
        None
    """
    if isinstance(break_by, str):
        break_by = [break_by]

    if filter_query is not None:
        data = df_res.query(filter_query).reset_index(drop=True).copy()
        txt_title = f"Accuracy by Operation Type\n{filter_query}"
    else:
        data = df_res.copy()
        txt_title = "Accuracy by Operation Type"

    # Set plot style and context
    sns.set_style("whitegrid")
    sns.set_context("paper")

    if break_by is None:
        # Create a single plot
        fig = create_single_plot(
            data, hue_group, txt_title, legend_loc, plot_type, figure_size
        )
    else:
        if len(break_by) == 1:
            # Create subplots 1D
            fig = create_subplots_1d(
                data, hue_group, break_by, txt_title, legend_loc, plot_type
            )
        elif len(break_by) == 2:
            # Create subplots 2D
            fig = create_subplots_2d(
                data, hue_group, break_by, txt_title, legend_loc, plot_type
            )

    if save_fig != "":
        fig.savefig(
            save_fig,
            bbox_inches="tight",
            backend="pdf",
        )
    return fig


def create_single_plot(
    data,
    hue_group,
    txt_title,
    legend_loc="upper right",
    plot_type="bar",
    figure_size=(8, 3),
):
    """
    Create a single bar plot.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data.
        hue_group (str): The column name to group the data by.
        txt_title (str): The title of the plot.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figure_size, dpi=600)
    if plot_type == "bar":
        sns.barplot(
            x=NAME_SET_OP,
            y=METRIC_TO_VISUALIZE,
            hue=hue_group,
            data=data,
            errorbar="ci",
            ax=ax,
        )
    elif plot_type == "violin":
        if hue_group is not None:
            sns.violinplot(
                x=NAME_SET_OP,
                y=METRIC_TO_VISUALIZE,
                hue=hue_group,
                data=data,
                inner="quartile",
                ax=ax,
            )
        else:
            sns.violinplot(
                x=NAME_SET_OP,
                y=METRIC_TO_VISUALIZE,
                data=data,
                inner="quartile",
                bw_adjust=0.5,
                ax=ax,
            )
    ax.set_xlabel(NAME_SET_OP)
    ax.set_ylabel(METRIC_TO_VISUALIZE)
    ax.legend(loc=legend_loc, title=hue_group, bbox_to_anchor=(1.18, 1.02))
    ax.set_title(txt_title)
    if plot_type == "bar":
        ax.set_ylim(-5, 100)
    elif plot_type == "violin":
        ax.set_ylim(0, None)
    plt.show()

    return fig


def create_subplots_1d(
    data,
    hue_group,
    break_by,
    txt_title,
    legend_loc="upper right",
    plot_type="bar",
):
    """
    Create subplots for each column in the specified list.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data.
        hue_group (str): The column name to group the data by.
        break_by (str): The column name to create subplots for.
        txt_title (str): The title of the plot.

    Returns:
        None
    """
    if isinstance(break_by, list):
        break_by = break_by[0]
    rows = list(data[break_by].unique())
    num_rows = len(rows)
    fig, axs = plt.subplots(num_rows, 1, figsize=(8, 3 * num_rows), dpi=200)

    for i, row in enumerate(rows):
        if num_rows > 1:
            ax = axs[i]
        else:
            ax = axs

        sns.barplot(
            x=NAME_SET_OP,
            y=METRIC_TO_VISUALIZE,
            hue=hue_group,
            data=data[data[break_by] == row],
            errorbar="ci",
            ax=ax,
        )
        ax.set_xlabel(NAME_SET_OP)
        ax.set_ylabel(METRIC_TO_VISUALIZE)
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=(1, 1),
            title=hue_group,
        )
        ax.set_title(f"{txt_title} ({row})")
        ax.set_ylim(-5, 100)

    plt.tight_layout()
    plt.show()

    return fig


def build_condition(break_by, row_val, col_val):
    """based on the data type, we create the condition"""
    row_cond = (
        f"({break_by[0]} == {row_val})"
        if isinstance(row_val, np.int64)
        else f"({break_by[0]} == ['{row_val}'])"
    )
    col_cond = (
        f"({break_by[1]} == {col_val})"
        if isinstance(col_val, np.int64)
        else f"({break_by[1]} == ['{col_val}'])"
    )
    return f"{row_cond} & {col_cond}"


def create_subplots_2d(
    data,
    hue_group,
    break_by,
    txt_title,
    legend_loc="upper right",
    plot_type="bar",
):
    """
    Create subplots for each combination of values in the specified list.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data.
        hue_group (str): The column name to group the data by.
        break_by (list): A list of two column names to create subplots for.
        txt_title (str): The title of the plot.

    Returns:
        None
    """

    if len(break_by) != 2:
        raise ValueError("break_by must be a list of two column names")

    rows = list(data[break_by[0]].unique())
    cols = list(data[break_by[1]].unique())
    num_rows = len(rows)
    num_cols = len(cols)

    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(8 * num_cols, 3 * num_rows),
        dpi=600,
        squeeze=False,
    )

    for i, row_val in enumerate(rows):
        for j, col_val in enumerate(cols):
            # condition = build_condition(break_by, row_val, col_val)
            # data_subset = data.query(condition)
            data_subset = data[
                (data[break_by[0]] == row_val)
                & (data[break_by[1]] == col_val)
            ]
            ax = axs[i, j]
            sns.barplot(
                x=NAME_SET_OP,
                y=METRIC_TO_VISUALIZE,
                hue=hue_group,
                data=data_subset,
                errorbar="ci",
                ax=ax,
            )
            ax.set_xlabel(NAME_SET_OP)
            ax.set_ylabel(METRIC_TO_VISUALIZE)
            # add title to legened
            ax.legend(
                loc=legend_loc,
                title=hue_group,
            )
            ax.set_title(
                f"{txt_title}\n"
                f"({break_by[0]}={row_val}, "
                f"{break_by[1]}={col_val})"
            )
            ax.set_ylim(-5, 100)

    plt.tight_layout()
    plt.show()

    return fig


def create_violin_agg(
    df_new,
    x_name,
    hue=None,
    figure_size=(3, 3),
    save_fig="",
    split_bar=True,
    fontsize=20,
    legend_loc="best",
    save_raw_experiment=True,
    supp_root=None,
):
    color_list = [
        "royalblue",
        "orange",
        "red",
        "yellow",
        "lightseagreen",
        "plum",
        "lightcoral",
        "olive",
        "chocolate",
    ]

    fig, ax = plt.subplots(figsize=figure_size, dpi=600)
    # make it dashed
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.5, linestyle="--")
    # reduce line width for grid
    # matplotlib.rcParams['axes.linewidth'] = 0.8
    sns.violinplot(
        x=x_name,
        y=METRIC_TO_VISUALIZE,
        data=df_new,
        hue=hue,
        split=split_bar,
        inner="quart",
        bw_adjust=0.3,
        linewidth=0.8,
        linecolor="k",
        palette=color_list,
        ax=ax,
    )

    # increase all font sizes for fig
    # increase legend font size and its title
    if hue is not None:
        if legend_loc == "outer right":
            ax.legend(
                fontsize=fontsize - 6, title_fontsize=fontsize - 2, title=hue
            )
            # outer location right
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.21, 1),
                title=hue,
            )
        else:
            ax.legend(
                fontsize=fontsize - 6,
                title_fontsize=fontsize - 2,
                title=hue,
                loc=legend_loc,
            )
    for ax in fig.axes:
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(fontsize)

    # set xlim
    ax.set_ylim(-5, 105)
    # set grid on, only y-axis
    plt.tight_layout()
    plt.show()

    if save_fig != "":
        fig.savefig(
            save_fig,
            bbox_inches="tight",
            backend="pdf",
        )

    if save_raw_experiment:
        if hue is not None:
            df_new.to_csv(
                os.path.join(supp_root, f"{x_name}_and_{hue}.csv"),
                index=False,
            )
        else:
            df_new.to_csv(
                os.path.join(supp_root, f"{x_name}.csv"), index=False
            )
    return fig
