# coding: utf-8

import argparse
import ast
import itertools
import logging
import os
import random
from collections.abc import Iterable
from itertools import product
from typing import Dict, List, Union

import pandas as pd
from tqdm import tqdm

from setlexsem.constants import PATH_PROMPTS_ROOT
from setlexsem.generate.generate_sets import get_sampler, make_hps_set
from setlexsem.generate.prompt import (
    PromptConfig,
    get_ground_truth,
    get_prompt,
)
from setlexsem.generate.sample import Sampler
from setlexsem.generate.utils_io import load_generated_data
from setlexsem.utils import get_prompt_file_path, read_config

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


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
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save data to disk",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite data",
    )
    return parser


def make_hps_prompt(
    op_list=None,
    k_shot=None,
    prompt_type=None,
    prompt_approach=None,
    is_fix_shot=None,
    config: Dict[str, List[Union[str, int]]] = {},
):
    """Make hyperparamters for the prompts"""
    if config:
        op_list = config["op_list"]
        k_shot = config.get("k_shot")
        prompt_type = config.get("prompt_type")
        prompt_approach = config.get("prompt_approach")
        is_fix_shot = config.get("is_fix_shot")

    # Wrap each parameter in a list if it isn’t already, to enable Cartesian product
    param_grid = {
        "op_list": op_list if isinstance(op_list, list) else [op_list],
        "k_shot": k_shot if isinstance(k_shot, list) else [k_shot],
        "prompt_type": (
            prompt_type if isinstance(prompt_type, list) else [prompt_type]
        ),
        "prompt_approach": (
            prompt_approach
            if isinstance(prompt_approach, list)
            else [prompt_approach]
        ),
        "is_fix_shot": (
            is_fix_shot if isinstance(is_fix_shot, list) else [is_fix_shot]
        ),
    }

    # Generate combinations of all parameters as dictionaries
    keys, values = zip(*param_grid.items())
    return (dict(zip(keys, v)) for v in product(*values))


def get_prompt_config(
    prompt_config: Dict[str, List[Union[str, int]]], k_shot_sampler: Sampler
):
    """Convert dictionary to PromptConfig class"""
    # prepare config for prompt
    prompt_config_ready = PromptConfig(
        operation=prompt_config["op_list"],
        k_shot=prompt_config["k_shot"],
        type=prompt_config["prompt_type"],
        approach=prompt_config["prompt_approach"],
        sampler=k_shot_sampler,
        is_fixed_shots=prompt_config["is_fix_shot"],
    )

    return prompt_config_ready


def create_prompts_from_sampler(
    sampler: Sampler,
    prompt_config: Dict[str, List[Union[str, int]]],
    k_shot_sampler: Sampler,
    num_runs=100,
    add_roles=False,  # Claude Instant
):
    """Create the prompt and the ground truth from Sampler and PromptConfig"""
    # get prompt config
    prompt_config_ready = get_prompt_config(prompt_config, k_shot_sampler)

    results = 0
    prompt_and_ground_truth = []
    for i in tqdm(range(num_runs)):
        try:
            # create two sets from the sampler
            if isinstance(sampler, Iterable):
                # get next set from generator
                A, B = next(sampler)
                A = ast.literal_eval(A)
                B = ast.literal_eval(B)
            else:
                # generate next set
                A, B = sampler()
        except:
            logger.warning(f"No set: {sampler}")
            continue

        try:
            # Assign operation to the prompt_config
            prompt = get_prompt(
                A,
                B,
                prompt_config_ready,
                add_roles=add_roles,
            )
        except:
            logger.warning(f"No prompt: {prompt_config}")
            continue

        ground_truth = get_ground_truth(prompt_config_ready.operation, A, B)

        prompt_and_ground_truth.append(
            {
                "prompt": prompt,
                "ground_truth": ground_truth,
                **prompt_config_ready.to_dict(),
            }
        )

    return prompt_and_ground_truth


def create_prompts(
    set_types=None,
    n=None,
    m_A=None,
    m_B=None,
    item_len=None,
    decile_group=None,
    swap_status=None,
    overlap_fraction=None,
    op_list=None,
    k_shot=None,
    prompt_type=None,
    prompt_approach=None,
    is_fix_shot=None,
    number_of_data_points=10,
    random_seed_value=292,
    add_roles=False,
    data_config: Dict[str, List[Union[str, int]]] = {},
    prompt_config: Dict[str, List[Union[str, int]]] = {},
):
    """Generate prompts for the given hyperparameters"""
    # generator for set construction
    if data_config:
        make_hps_generator = make_hps_set(config=data_config)
    else:
        make_hps_generator = make_hps_set(
            config={
                "set_types": set_types,
                "n": n,
                "m_A": m_A,
                "m_B": m_B,
                "item_len": item_len,
                "decile_group": decile_group,
                "swap_status": swap_status,
                "overlap_fraction": overlap_fraction,
            }
        )

    # generator for prompts
    if prompt_config:
        make_hps_prompt_generator = make_hps_prompt(config=prompt_config)
    else:
        make_hps_prompt_generator = make_hps_prompt(
            config={
                "op_list": op_list,
                "k_shot": k_shot,
                "prompt_type": prompt_type,
                "prompt_approach": prompt_approach,
                "is_fix_shot": is_fix_shot,
            }
        )

    make_hps_generator, make_hps_generator_copy = itertools.tee(
        make_hps_generator
    )

    # report number of overall experiments
    n_experiments = len(list(make_hps_prompt_generator)) * len(
        list(make_hps_generator_copy)
    )

    # go through hyperparameters and run the experiment
    counter_exp = 1
    output = {}
    for hp in make_hps_generator:
        # generator for prompts
        if prompt_config:
            make_hps_prompt_generator = make_hps_prompt(config=prompt_config)
        else:
            make_hps_prompt_generator = make_hps_prompt(
                config={
                    "op_list": op_list,
                    "k_shot": k_shot,
                    "prompt_type": prompt_type,
                    "prompt_approach": prompt_approach,
                    "is_fix_shot": is_fix_shot,
                }
            )

        for hp_prompt in make_hps_prompt_generator:
            # Initilize Seed for each combination
            random_state = random.Random(random_seed_value)

            # Create Sampler()
            try:
                sampler = get_sampler(hp, random_state)
            except Exception as e:
                logger.warning(f"No sampler: {hp} | {e}")
                counter_exp += 1
                continue

            # create kshot sampler, before loading data
            k_shot_sampler = sampler.create_sampler_for_k_shot()

            # Create prompts
            try:
                prompt_and_ground_truth = create_prompts_from_sampler(
                    sampler,
                    prompt_config=hp_prompt,
                    k_shot_sampler=k_shot_sampler,
                    num_runs=number_of_data_points,
                    add_roles=add_roles,  # Claude Instant
                )
            except Exception as e:
                logger.warning(f"No prompt from sampler: {hp_prompt} | {e}")
                counter_exp += 1
                continue

            output[counter_exp] = prompt_and_ground_truth
            counter_exp += 1

    return output


def main(config_file, save_data, overwrite):
    # TODO: Add this to Config
    SWAP_STATUS = False
    add_roles = False

    # Read Config File and Assign Variables
    config = read_config(config_file)

    # Experiment config
    number_of_data_points = config["N_RUN"]

    RANDOM_SEED_VAL = config["RANDOM_SEED_VAL"]
    OP_LIST = config["OP_LIST"]

    # Sampler/Sets Config
    SET_TYPES = config["SET_TYPES"]
    N = replace_none(config["N"])
    M_A = replace_none(config["M_A"])
    M_B = replace_none(config["M_B"])
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
    make_hps_prompt_generator = make_hps_prompt(
        config={
            "op_list": OP_LIST,
            "k_shot": K_SHOT,
            "prompt_type": PROMPT_TYPE,
            "prompt_approach": PROMPT_APPROACH,
            "is_fix_shot": IS_FIX_SHOT,
        }
    )
    # generator for set construction
    make_hps_generator = make_hps_set(
        config={
            "set_types": SET_TYPES,
            "n": N,
            "m_A": M_A,
            "m_B": M_B,
            "item_len": ITEM_LEN,
            "decile_group": DECILE_NUM,
            "swap_status": SWAP_STATUS,
            "overlap_fraction": OVERLAP_FRACTION,
        }
    )
    make_hps_generator, make_hps_generator_copy = itertools.tee(
        make_hps_generator
    )

    # report number of overall experiments
    n_experiments = len(list(make_hps_prompt_generator)) * len(
        list(make_hps_generator_copy)
    )
    logger.info(f"Creating prompts for {n_experiments} configurations...")

    # go through hyperparameters and run the experiment
    counter_exp = 1
    for hp_set in make_hps_generator:
        make_hps_prompt_generator = make_hps_prompt(
            config={
                "op_list": OP_LIST,
                "k_shot": K_SHOT,
                "prompt_type": PROMPT_TYPE,
                "prompt_approach": PROMPT_APPROACH,
                "is_fix_shot": IS_FIX_SHOT,
            }
        )
        for hp_prompt in make_hps_prompt_generator:
            logger.info(f"--- Prompt #{counter_exp} out of {n_experiments}")

            # Initilize Seed for each combination
            random_state = random.Random(RANDOM_SEED_VAL)

            # Create Sampler()
            try:
                sampler = get_sampler(hp_set, random_state)
            except Exception as e:
                logger.error(f"------> Error: Skipping this experiment: {e}")
                counter_exp += 1
                continue
            logger.info(sampler)

            # create kshot sampler, before loading data
            k_shot_sampler = sampler.create_sampler_for_k_shot()

            # load already created data
            sampler = load_generated_data(sampler, RANDOM_SEED_VAL)

            # Create prompts
            try:
                prompt_and_ground_truth = create_prompts_from_sampler(
                    sampler,
                    prompt_config=hp_prompt,
                    k_shot_sampler=k_shot_sampler,
                    num_runs=number_of_data_points,
                    add_roles=add_roles,  # Claude Instant
                )
            except Exception as e:
                logger.error(f"------> Error: Skipping this experiment: {e}")
                counter_exp += 1
                continue

            counter_exp += 1

            # create path based on hp and hp_prompt
            folder_structure, filename = get_prompt_file_path(
                hp_set, hp_prompt, RANDOM_SEED_VAL
            )
            path_to_prompts = os.path.join(
                PATH_PROMPTS_ROOT, folder_structure, filename
            )
            if save_data:
                os.makedirs(os.path.dirname(path_to_prompts), exist_ok=True)
                # save prompts
                if not os.path.exists(path_to_prompts) or overwrite:
                    pd.DataFrame(prompt_and_ground_truth).fillna(
                        "None"
                    ).to_csv(path_to_prompts, index=False)
                else:
                    logger.info(
                        f"Prompt file already exists, skipping: {folder_structure}/{filename}"
                    )

    logger.info("Done!")


# init
if __name__ == "__main__":
    # parse args
    parser = get_parser()
    args = parser.parse_args()
    config_path = args.config_path
    save_data = args.save_data
    overwrite = args.overwrite

    main(config_path, save_data, overwrite)
