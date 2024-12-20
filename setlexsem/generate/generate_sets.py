import argparse
import itertools
import logging
import random
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from setlexsem.generate.sample import (
    BasicNumberSampler,
    BasicWordSampler,
    DeceptiveWordSampler,
    DecileWordSampler,
    OverlapSampler,
    Sampler,
)
from setlexsem.generate.utils_io import save_generated_sets

# add logger and line number
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


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
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite data"
    )
    return parser


def make_sets_from_sampler(
    sample_set: Sampler,
    num_runs: int,
) -> List[Dict[str, Any]]:
    """Generate random sets from the sampler"""

    # initlize the dataset
    set_list = []
    for i in range(num_runs):
        try:
            # create two sets from the sampler
            A, B = sample_set()
            # loop through operations (on the same random sets)
            set_list.append(
                {
                    "experiment_run": i,
                    "A": A,
                    "B": B,
                }
            )
        except:
            continue
    return set_list


def read_config_make_sets(config_path: str = "config.yaml"):
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


def make_hps_set(
    set_types=None,
    n=None,
    m_A=None,
    m_B=None,
    item_len=None,
    decile_group=None,
    swap_status=None,
    overlap_fraction=None,
    config: Dict[str, Any] = {},
) -> Iterable:
    if config:
        set_types = config["set_types"]
        n = config.get("n")
        m_A = config.get("m_A")
        m_B = config.get("m_B")
        item_len = config.get("item_len")
        decile_group = config.get("decile_group")
        swap_status = config.get("swap_status")
        overlap_fraction = config.get("overlap_fraction")

    # Wrap each parameter in a list if it isn’t already, to enable Cartesian product
    param_grid = {
        "set_type": set_types if isinstance(set_types, list) else [set_types],
        "n": n if isinstance(n, list) else [n],
        "m_A": m_A if isinstance(m_A, list) else [m_A],
        "m_B": m_B if isinstance(m_B, list) else [m_B],
        "item_len": item_len if isinstance(item_len, list) else [item_len],
        "decile_group": (
            decile_group if isinstance(decile_group, list) else [decile_group]
        ),
        "swap_status": (
            swap_status if isinstance(swap_status, list) else [swap_status]
        ),
        "overlap_fraction": (
            overlap_fraction
            if isinstance(overlap_fraction, list)
            else [overlap_fraction]
        ),
    }

    # Generate combinations of all parameters as dictionaries
    keys, values = zip(*param_grid.items())
    return (dict(zip(keys, v)) for v in product(*values))


def get_sampler(hp: Dict[str, Any], random_state: random.Random) -> Sampler:
    set_type = hp["set_type"]

    if hp["set_type"] == "numbers":
        sampler = BasicNumberSampler(
            n=hp["n"],
            m_A=hp["m_A"],
            m_B=hp["m_B"],
            item_len=hp.get("item_len"),
            random_state=random_state,
        )
    elif set_type == "words":
        sampler = BasicWordSampler(
            n=hp["n"],
            m_A=hp["m_A"],
            m_B=hp["m_B"],
            item_len=hp.get("item_len"),
            random_state=random_state,
        )
    elif "deciles" in set_type:
        sampler = DecileWordSampler(
            n=hp["n"],
            m_A=hp["m_A"],
            m_B=hp["m_B"],
            item_len=hp.get("item_len"),
            decile_num=hp.get("decile_group"),
            random_state=random_state,
        )
    elif set_type == "deceptive_words":
        sampler = DeceptiveWordSampler(
            n=hp["n"],
            m_A=hp["m_A"],
            m_B=hp["m_B"],
            random_state=random_state,
            swap_set_elements=hp.get("swap_status"),
            swap_n=hp["m_A"] // 2,  # TODO: Change this to be a parameter
        )

    # create overlapping sets
    if hp["overlap_fraction"] is not None:
        sampler = OverlapSampler(
            sampler, overlap_fraction=hp.get("overlap_fraction")
        )
    return sampler


def make_sets(
    set_types=None,
    n=None,
    m_A=None,
    m_B=None,
    item_len=None,
    decile_group=None,
    swap_status=None,
    overlap_fraction=None,
    config: Dict[str, Any] = {},
    number_of_data_points: int = 100,
    seed_value: int = 292,
    random_state=None,
) -> Tuple[Dict[Any, Any], Sampler]:
    if config:
        set_types = config["set_type"]
        n = config.get("n")
        m_A = config.get("m_A")
        m_B = config.get("m_B")
        item_len = config.get("item_len")
        decile_group = config.get("decile_group")
        swap_status = config.get("swap_status")
        overlap_fraction = config.get("overlap_fraction")

    make_hps_generator = make_hps_set(
        set_types,
        n,
        m_A,
        m_B,
        item_len,
        decile_group,
        swap_status,
        overlap_fraction,
    )
    all_sets = []
    for hp in make_hps_generator:
        # random_state = random.Random(seed_value)

        try:
            sampler = get_sampler(hp, random_state)

            # get synthetic sets
            synthetic_sets = make_sets_from_sampler(
                sample_set=sampler, num_runs=number_of_data_points
            )
        except:
            logger.warning(f"No data: {hp}")
            continue

        temp_hp = hp
        for ds in synthetic_sets:
            temp_hp.update(ds)
            all_sets.append(temp_hp)

    return all_sets


# if __name__ == "__main__":
#     # parse args
#     parser = get_parser()
#     args = parser.parse_args()
#     config_path = args.config_path
#     save_data = args.save_data
#     number_of_data_points = args.number_of_data_points
#     seed_value = args.seed_value
#     overwrite = args.overwrite

#     # read config file
#     config = read_config_make_sets(config_path=config_path)

#     # make hyperparameters
#     make_hps_generator = make_hps_set(config=config)
#     make_hps_generator, make_hps_generator_copy = itertools.tee(
#         make_hps_generator
#     )
#     n_configurations = len(list(make_hps_generator_copy))
#     logger.info(f"Creating sets for {n_configurations} configurations...")

#     for hp_set in make_hps_generator:
#         random_state = random.Random(seed_value)
#         try:
#             sampler = get_sampler(hp_set, random_state)

#             synthetic_sets = make_sets_from_sampler(
#                 sample_set=sampler, num_runs=number_of_data_points
#             )

#             logger.info(f"Generated {sampler}")
#             if save_data:
#                 save_generated_sets(
#                     synthetic_sets,
#                     sampler,
#                     seed_value,
#                     number_of_data_points,
#                     overwrite=overwrite,
#                 )

#         except Exception as e:
#             logger.warning(f"Skipping: {e} / {sampler}")
#             continue

#     logger.info("Dataset is created!")
