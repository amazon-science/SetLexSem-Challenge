import argparse
import itertools
import logging
import random
from itertools import product
from typing import Any, Dict, Iterable, List

import yaml

from setlexsem.generate.sample import (
    BasicNumberSampler,
    BasicWordSampler,
    DeceptiveWordSampler,
    DecileWordSampler,
    OverlapSampler,
    Sampler,
)
from setlexsem.generate.utils_data_generation import save_generated_data


# define argparser
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to config file for generating data",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save data to disk",
    )
    parser.add_argument("--number-of-data-points", type=int, default=10000)
    parser.add_argument("--seed-value", type=int, default=292)
    return parser


def generate_data_from_sampler(
    sampler: Sampler,
    num_runs: int,
):
    """Generate random data from the sampler"""
    # create the dataset
    set_list = []
    for i in range(num_runs):
        # create two sets from the sampler
        A, B = sampler()
        # loop through operations (on the same random sets)
        set_list.append(
            {
                "experiment_run": i,
                "A": A,
                "B": B,
            }
        )

    return set_list


def read_data_gen_config(config_path="config.yaml"):
    """Read config file from YAML"""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


def make_hps(
    set_types=None,
    n=None,
    m=None,
    item_len=None,
    decile_group=None,
    swap_status=None,
    overlap_fraction=None,
    config: Dict = {},
):
    if config:
        set_types = config["set_types"]
        n = config.get("n")
        m = config.get("m")
        item_len = config.get("item_len")
        decile_group = config.get("decile_group")
        swap_status = config.get("swap_status")
        overlap_fraction = config.get("overlap_fraction")

    # Wrap each parameter in a list if it isn’t already, to enable Cartesian product
    param_grid = {
        "set_type": set_types if isinstance(set_types, list) else [set_types],
        "n": n if isinstance(n, list) else [n],
        "m": m if isinstance(m, list) else [m],
        "item_len": item_len if isinstance(item_len, list) else [item_len],
        "decile_group": decile_group
        if isinstance(decile_group, list)
        else [decile_group],
        "swap_status": swap_status
        if isinstance(swap_status, list)
        else [swap_status],
        "overlap_fraction": overlap_fraction
        if isinstance(overlap_fraction, list)
        else [overlap_fraction],
    }

    # Generate combinations of all parameters as dictionaries
    keys, values = zip(*param_grid.items())
    return (dict(zip(keys, v)) for v in product(*values))


def get_sampler(hp, random_state):
    set_type = hp["set_type"]

    if set_type == "numbers":
        sampler = BasicNumberSampler(
            n=hp["n"],
            m=hp["m"],
            item_len=hp.get("item_len"),
            random_state=random_state,
        )
    elif set_type == "words":
        sampler = BasicWordSampler(
            n=hp["n"],
            m=hp["m"],
            item_len=hp.get("item_len"),
            random_state=random_state,
        )
    elif set_type == "deceptive_words":
        sampler = DeceptiveWordSampler(
            n=hp["n"],
            m=hp["m"],
            random_state=random_state,
            swap_set_elements=hp.get("swap_status"),
            swap_n=hp["m"] // 2,
        )
    elif set_type == "deciles":
        sampler = DecileWordSampler(
            n=hp["n"],
            m=hp["m"],
            item_len=hp.get("item_len"),
            decile_num=hp.get("decile_group"),
        )

    # create overlapping sets
    if hp["overlap_fraction"] is not None:
        sampler = OverlapSampler(
            sampler, overlap_fraction=hp.get("overlap_fraction")
        )

    return sampler


def generate_data(
    set_types=None,
    n=None,
    m=None,
    item_len=None,
    decile_group=None,
    swap_status=None,
    overlap_fraction=None,
    config: Dict = {},
    number_of_data_points=100,
    seed_value=292,
):
    if config:
        set_types = config["set_types"]
        n = config.get("n")
        m = config.get("m")
        item_len = config.get("item_len")
        decile_group = config.get("decile_group")
        swap_status = config.get("swap_status")
        overlap_fraction = config.get("overlap_fraction")

    make_hps_generator = make_hps(
        set_types, n, m, item_len, decile_group, swap_status, overlap_fraction
    )
    output = {}
    for hp in make_hps_generator:
        random_state = random.Random(seed_value)

        try:
            sampler = get_sampler(hp, random_state)

            dict_gen_data = generate_data_from_sampler(
                sampler=sampler, num_runs=number_of_data_points
            )
        except:
            continue

        output[tuple(hp.items())] = dict_gen_data

    return output


if __name__ == "__main__":
    # parse args
    parser = get_parser()
    args = parser.parse_args()
    config_path = args.config_path
    save_data = args.save_data
    number_of_data_points = args.number_of_data_points
    seed_value = args.seed_value
    overwrite = args.overwrite

    # read config file
    config = read_data_gen_config(config_path=config_path)

    # add logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    make_hps_generator = make_hps(config=config)
    make_hps_generator, make_hps_generator_copy = itertools.tee(
        make_hps_generator
    )
    n_experiments = len(list(make_hps_generator_copy))
    logger.info(f"Experiment will run for {n_experiments} times")

    for hp in make_hps_generator:
        random_state = random.Random(seed_value)
        try:
            sampler = get_sampler(hp, random_state)

            dict_gen_data = generate_data_from_sampler(
                sampler=sampler, num_runs=number_of_data_points
            )

            logger.info(f"Generated {sampler}")
            if save_data:
                save_generated_data(
                    dict_gen_data,
                    sampler,
                    seed_value,
                    number_of_data_points,
                    overwrite=overwrite,
                )

        except Exception as e:
            logger.warning(f"skipping: {e} / {sampler}")
            continue

    logger.info("Dataset is complete!")
