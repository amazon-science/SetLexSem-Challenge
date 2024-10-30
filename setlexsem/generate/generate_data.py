import argparse
import json
import logging
import random
from itertools import product

from setlexsem.generate.sample import (
    BasicNumberSampler,
    BasicWordSampler,
    DeceptiveWordSampler,
    DecileWordSampler,
    OverlapSampler,
)
from setlexsem.generate.utils_data_generation import (
    generate_data,
    load_generated_data,
    save_generated_data,
)


# define argparser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_files",
        type=int,
        help="save files to disk",
    )
    parser.add_argument("--number_of_data_points", type=int, default=10000)
    parser.add_argument("--seed_value", type=int, default=292)
    args = parser.parse_args()
    return args


def read_data_gen_config():
    # TODO: fix this
    config = {
        "set_types": ["deciles"],
        "n": [10000],
        "m": [2, 4, 8, 16],
        "item_len": [5],
        "overlap_fraction": [None],
        "decile_group": list(range(1, 10)),
        "swap_status": [False],
    }

    return config


if __name__ == "__main__":
    # parse args
    args = parse_args()
    SAVE_DATA = args.save_files
    NUM_RUNS = args.number_of_data_points
    SEED_VALUE = args.seed_value

    # read config file
    config = read_data_gen_config()

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
            if SAVE_DATA:
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
