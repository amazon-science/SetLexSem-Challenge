# coding: utf-8

import logging
import os

import pandas as pd

from setlexsem.constants import PATH_DATA_ROOT
from setlexsem.generate.sample import Sampler
from setlexsem.utils import get_data_filename

# define the logger
logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.INFO)


def save_generated_sets(
    set_list,
    sampler: Sampler,
    random_seed: int,
    num_runs: int,
    overwrite=False,
    rename_sampler=None,
):
    """Save generated data from the sampler"""
    # prepare filenames and check if the file exist
    filename = get_data_filename(
        sampler.make_filename(), random_seed, num_runs
    )

    # convert to dataframe
    df_data = pd.DataFrame(set_list)

    # prepare folder structure
    path_save_folder = os.path.join(
        PATH_DATA_ROOT, sampler.get_members_type()
    )
    if not os.path.exists(path_save_folder):
        os.makedirs(path_save_folder)

    # prepare path to filename
    path_data = os.path.join(path_save_folder, filename)

    # save data if it does not exist or we are overwriting
    if not os.path.exists(path_data) or overwrite:
        df_data.to_csv(path_data, index=False)
        LOGGER.info(f"Saving results to {path_data}")
    else:
        LOGGER.info(f"Data already exists at {path_data}, skipping...")


def load_generated_data(
    sampler: Sampler, random_seed, num_runs_data_stored_at=10000
):
    """Load generated data from the sampler as a generator iterator"""
    # prepare filenames and check if the file exist
    filename = get_data_filename(
        sampler.make_filename(), random_seed, num_runs_data_stored_at
    )

    # prepare path to filename
    path_data = os.path.join(
        PATH_DATA_ROOT, sampler.get_members_type(), filename
    )

    # check if the file exists
    if os.path.exists(path_data):
        df_data = pd.read_csv(path_data)
        generator = ((row["A"], row["B"]) for _, row in df_data.iterrows())
        return iter(generator)
    else:
        LOGGER.error(f"Data not found at {path_data}, skipping...")
        return iter([])  # Return an empty iterator
