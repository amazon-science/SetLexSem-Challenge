import argparse
import ast
import itertools
import logging
import random
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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


def astype_set(raw_input):
    if not isinstance(raw_input, set):
        if type(raw_input) == str:
            raw_input = raw_input.strip()
            set_out = ast.literal_eval(raw_input)
        else:
            set_out = set(raw_input)

        return set_out
    else:
        return set_out


def parse_set_pair(raw_set_a: str, raw_set_b: str) -> Tuple[set, set]:
    """Parse string representations of sets into Python sets"""
    try:
        set_a = astype_set(raw_set_a)
        set_b = astype_set(raw_set_b)
        return set_a, set_b
    except (ValueError, SyntaxError) as e:
        logger.error(f"Failed to parse sets: {e}")
        raise


def generate_set_pair(sampler: Union[Iterable, callable]) -> Tuple[set, set]:
    """Generate a pair of sets from either an iterable or callable sampler.

    This function is designed to provide a flexible interface for generating set pairs.

    The function's flexibility allows users to provide their own custom sampling logic,
    either as an iterable or a callable, making it adaptable to different use cases and
    data sources.

    Notes:
        - If sampler is an Iterable, it should yield raw set pairs that will be parsed.
        - If sampler is a callable, it should directly return a tuple of two sets.
        - Any exceptions during generation are caught and logged as warnings.

    Examples:
        Using an iterable:
        >>> def pair_generator():
        ...     yield ([1, 2, 3], [3, 4, 5])
        ...     yield ([2, 4, 6], [1, 3, 5])
        >>> set_a, set_b = generate_set_pair(pair_generator())
        >>> print(set_a, set_b)
        {1, 2, 3} {3, 4, 5}

        Using a callable [or, Sampler class]:
        >>> import random
        >>> def random_set_pair():
        ...     return set(random.sample(range(10), 3)), set(random.sample(range(10), 3))
        >>> set_a, set_b = generate_set_pair(random_set_pair)
        >>> print(set_a, set_b)  # Output will vary due to randomness
        {1, 4, 7} {2, 5, 9}
    """
    try:
        if isinstance(sampler, Iterable):
            raw_a, raw_b = next(sampler)
            return parse_set_pair(raw_a, raw_b)
        else:
            return sampler()
    except Exception as e:
        # to reduce warnings, we catch all these and surface at the end
        logger.debug(f"Failed to generate set pair from sampler: {e}")
        return None, None


def make_sets_from_sampler(
    sample_set: Sampler,
    num_runs: int,
) -> List[Dict[str, Any]]:
    """Generate random sets from the sampler"""

    # initlize the dataset
    empty_sample_count = 0
    set_list = []
    for i in range(num_runs):
        # create two sets from the sampler
        A, B = generate_set_pair(sample_set)
        if A is None or B is None:
            empty_sample_count += 1
            continue

        # loop through operations (on the same random sets)
        set_list.append(
            {
                "experiment_run": i,
                "A": A,
                "B": B,
            }
        )

    if empty_sample_count:
        logger.warning(
            f"Did not sample for `{empty_sample_count} out of {num_runs} cases` because set hyperparameter conditions could not be met."
        )

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
        "set_types": set_types
        if isinstance(set_types, list)
        else [set_types],
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
    set_type = hp["set_types"]
    if set_type == "numbers":
        sampler = BasicNumberSampler(
            n=hp.get("n"),
            m_A=hp["m_A"],
            m_B=hp["m_B"],
            item_len=hp.get("item_len"),
            random_state=random_state,
        )
    elif set_type == "words":
        sampler = BasicWordSampler(
            m_A=hp["m_A"],
            m_B=hp["m_B"],
            item_len=hp.get("item_len"),
            random_state=random_state,
        )
    elif "deciles" in set_type:
        sampler = DecileWordSampler(
            m_A=hp["m_A"],
            m_B=hp["m_B"],
            item_len=hp.get("item_len"),
            decile_num=hp.get("decile_group"),
            random_state=random_state,
        )
    elif set_type == "deceptive_words":
        sampler = DeceptiveWordSampler(
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
) -> Tuple[Dict[Any, Any], Sampler]:
    if config:
        set_types = config["set_types"]
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
    for hp_set in make_hps_generator:
        random_state = random.Random(seed_value)

        try:
            sampler = get_sampler(hp_set, random_state)

            # get synthetic sets
            synthetic_sets = make_sets_from_sampler(
                sample_set=sampler, num_runs=number_of_data_points
            )
        except:
            logger.warning(
                f"No data for: {hp_set} - Make sure hyperparameters can be used to generate a set."
            )
            continue

        # add hyperparameters and concatenate results
        for ds in synthetic_sets:
            temp_hp = hp_set.copy()
            temp_hp.update(ds)
            all_sets.append(temp_hp)

    return all_sets


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
    config = read_config_make_sets(config_path=config_path)

    # make hyperparameters
    make_hps_generator = make_hps_set(config=config)
    make_hps_generator, make_hps_generator_copy = itertools.tee(
        make_hps_generator
    )
    n_configurations = len(list(make_hps_generator_copy))
    logger.info(f"Creating sets for {n_configurations} configurations...")

    for hp_set in make_hps_generator:
        random_state = random.Random(seed_value)
        try:
            sampler = get_sampler(hp_set, random_state)

            synthetic_sets = make_sets_from_sampler(
                sample_set=sampler, num_runs=number_of_data_points
            )

            logger.info(f"Generated {sampler}")
            if save_data:
                save_generated_sets(
                    synthetic_sets,
                    sampler,
                    seed_value,
                    number_of_data_points,
                    overwrite=overwrite,
                )

        except Exception as e:
            logger.warning(f"Skipping: {e} / {sampler}")
            continue

    logger.info("Dataset is created!")
