# coding: utf-8

import argparse
import ast
import logging
import os
import random
from collections.abc import Iterable
from itertools import product

import pandas as pd
from tqdm import tqdm

from setlexsem.constants import PATH_PROMPTS_ROOT
from setlexsem.generate.prompt import (
    PromptConfig,
    get_ground_truth,
    get_prompt,
)
from setlexsem.generate.sample import (
    BasicNumberSampler,
    BasicWordSampler,
    DeceptiveWordSampler,
    DecileWordSampler,
    OverlapSampler,
    Sampler,
)
from setlexsem.generate.utils_data_generation import load_generated_data
from setlexsem.utils import get_prompt_file_path, read_config


def replace_none(list_in):
    return [None if x == "None" else x for x in list_in]


# define argparser
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to config file for generating prompts",
    )
    return parser


def create_prompt(
    sampler: Sampler,
    prompt_config: PromptConfig,
    num_runs=100,
    add_roles=False,  # Claude Instant
):
    prompt_and_ground_truth = []
    for i in tqdm(range(num_runs)):
        # create two sets from the sampler
        if isinstance(sampler, Iterable):
            # get next set from generator
            A, B = next(sampler)
            A = ast.literal_eval(A)
            B = ast.literal_eval(B)
        else:
            # generate next set
            A, B = sampler()

        # Assign operation to the prompt_config
        prompt = get_prompt(
            A,
            B,
            prompt_config,
            add_roles=add_roles,
        )

        ground_truth = get_ground_truth(prompt_config.operation, A, B)

        prompt_and_ground_truth.append(
            {
                "prompt": prompt,
                "ground_truth": ground_truth,
                **prompt_config.to_dict(),
            }
        )

    return prompt_and_ground_truth


def main(config_file):
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(level=logging.INFO)

    # TODO: Add this to Config
    SWAP_STATUS = False
    add_roles = False

    # Read Config File and Assign Variables
    config = read_config(config_file)

    # Experiment config
    RANDOM_SEED_VAL = config["RANDOM_SEED_VAL"]
    OP_LIST = config["OP_LIST"]

    # Sampler/Sets Config
    SET_TYPES = config["SET_TYPES"]
    N = replace_none(config["N"])
    M = replace_none(config["M"])
    ITEM_LEN = replace_none(config["ITEM_LEN"])
    OVERLAP_FRACTION = replace_none(config["OVERLAP_FRACTION"])
    DECILE_NUM = replace_none(config["DECILE_NUM"])

    assert (
        len(OVERLAP_FRACTION) == 1
    ), "the code only runs on 1 overlap fraction value"

    # Prompt Config
    K_SHOT = config["K_SHOT"]
    PROMPT_TYPE = config["PROMPT_TYPE"]
    PROMPT_APPROACH = config["PROMPT_APPROACH"]
    IS_FIX_SHOT = config["IS_FIX_SHOT"]

    # generator for prompts
    def make_hps_prompt():
        return product(
            OP_LIST, K_SHOT, PROMPT_TYPE, PROMPT_APPROACH, IS_FIX_SHOT
        )

    # generator for set construction
    if DECILE_NUM[0] is not None:

        def make_hps():
            return product(
                SET_TYPES, N, M, ITEM_LEN, DECILE_NUM, OVERLAP_FRACTION
            )

    else:

        def make_hps():
            return product(SET_TYPES, N, M, ITEM_LEN, OVERLAP_FRACTION)

    # report number of overall experiments
    n_experiments = len(list(make_hps())) * len(list(make_hps_prompt()))
    LOGGER.info(f"Experiment will run for {n_experiments} times")

    # go through hyperparameters and run the experiment
    counter_exp = 1
    for hp in make_hps():
        for hp_prompt in make_hps_prompt():
            LOGGER.info(
                f"-------- EXPERIMENT #{counter_exp} out of {n_experiments}"
            )

            # Initilize Seed for each combination
            random_state = random.Random(RANDOM_SEED_VAL)

            # Create Sampler()
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
                    swap_set_elements=SWAP_STATUS,
                    swap_n=hp[2] // 2,
                )
            elif hp[0] == "decile_words":
                sampler = DecileWordSampler(
                    n=hp[1], m=hp[2], item_len=hp[3], decile_num=hp[4]
                )

            # if overlapping, create OverlapSampler()
            try:
                if "overlapping" in hp[0]:
                    sampler = OverlapSampler(sampler, overlap_fraction=hp[4])
            except:
                LOGGER.error("------> Error: Skipping this experiment")
                counter_exp += 1
                continue
            LOGGER.info(sampler)
            # create k-shot sampler
            k_shot_sampler = sampler.create_sampler_for_k_shot()

            # load already created data
            sampler = load_generated_data(sampler, RANDOM_SEED_VAL)

            # Create Prompt Config
            prompt_config = PromptConfig(
                operation=hp_prompt[0],
                k_shot=hp_prompt[1],
                type=hp_prompt[2],
                approach=hp_prompt[3],
                sampler=k_shot_sampler,
                is_fixed_shots=hp_prompt[4],
            )
            LOGGER.info(prompt_config)

            # Create prompts
            try:
                prompt_and_ground_truth = create_prompt(
                    sampler,
                    prompt_config,
                    num_runs=100,
                    add_roles=add_roles,  # Claude Instant
                )
            except Exception as e:
                LOGGER.error(f"------> Error: Skipping this experiment: {e}")
                counter_exp += 1
                continue

            counter_exp += 1

            # create path based on hp and hp_prompt
            folder_structure, filename = get_prompt_file_path(
                hp, hp_prompt, RANDOM_SEED_VAL
            )
            path_to_prompts = os.path.join(
                PATH_PROMPTS_ROOT, folder_structure, filename
            )
            os.makedirs(os.path.dirname(path_to_prompts), exist_ok=True)
            # save prompts
            pd.DataFrame(prompt_and_ground_truth).to_csv(
                path_to_prompts, index=False
            )

    LOGGER.info("Done!")


# init
if __name__ == "__main__":
    # parse args
    parser = get_parser()
    args = parser.parse_args()
    config_path = args.config_path

    main(config_path)
