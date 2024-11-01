import argparse
import logging
import random
from itertools import product

import yaml

from setlexsem.generate.sample import (
    BasicNumberSampler,
    BasicWordSampler,
    DeceptiveWordSampler,
    DecileWordSampler,
    OverlapSampler,
)
from setlexsem.generate.utils_data_generation import (
    generate_data,
    save_generated_data,
)


# define argparser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="path to config file for data generation",
    )
    parser.add_argument(
        "--save_data",
        type=int,
        required=True,
        help="save data to disk",
    )
    parser.add_argument("--number_of_data_points", type=int, default=10000)
    parser.add_argument("--seed_value", type=int, default=292)
    args = parser.parse_args()
    return args


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


if __name__ == "__main__":
    # parse args
    args = parse_args()
    config_path = args.config_path
    save_data = args.save_data
    NUM_RUNS = args.number_of_data_points
    SEED_VALUE = args.seed_value

    # read config file
    config = read_data_gen_config(config_path=config_path)

    # add logger
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(level=logging.INFO)

    if config["set_types"][0] == "deceptive_words":

        def make_hps():
            return product(
                config["set_types"],
                config["n"],
                config["m"],
                config["swap_status"],
            )

    elif config["set_types"][0] == "deciles":

        def make_hps():
            return product(
                config["set_types"],
                config["n"],
                config["m"],
                config["item_len"],
                config["decile_group"],
                config["overlap_fraction"],
            )

    else:

        def make_hps():
            return product(
                config["set_types"],
                config["n"],
                config["m"],
                config["item_len"],
                config["overlap_fraction"],
            )  # this is a generator

    n_experiments = len(list(make_hps()))
    print(f"Experiment will run for {n_experiments} times")

    # hp-0 set type | hp-1 N | hp-2 M | hp-3 item_len
    for hp in make_hps():
        random_state = random.Random(SEED_VALUE)

        try:
            if hp[0] == "numbers" or "BasicNumberSampler" in hp[0]:
                sampler = BasicNumberSampler(
                    n=hp[1],
                    m=hp[2],
                    item_len=hp[3],
                    random_state=random_state,
                )
            elif hp[0] == "words" or "BasicWordSampler" in hp[0]:
                sampler = BasicWordSampler(
                    n=hp[1],
                    m=hp[2],
                    item_len=hp[3],
                    random_state=random_state,
                )
            elif hp[0] == "deceptive_words":
                sampler = DeceptiveWordSampler(
                    n=hp[1],
                    m=hp[2],
                    random_state=random_state,
                    swap_set_elements=hp[3],
                    swap_n=hp[2] // 2,
                )

            elif hp[0] == "deciles":
                sampler = DecileWordSampler(
                    n=hp[1], m=hp[2], item_len=hp[3], decile_num=hp[4]
                )

            # add overlapping
            if "overlapping" in hp[0]:
                sampler = OverlapSampler(sampler, overlap_fraction=hp[4])

            dict_gen_data = generate_data(sampler=sampler, num_runs=NUM_RUNS)

            LOGGER.info(f"Generated {sampler}")
            if save_data:
                save_generated_data(
                    dict_gen_data,
                    sampler,
                    SEED_VALUE,
                    NUM_RUNS,
                    overwrite=False,
                )

        except Exception as e:
            LOGGER.warning(f"skipping: {e} / {sampler}")
            continue
