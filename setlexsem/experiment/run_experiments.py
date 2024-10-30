import argparse
import ast
import logging
import os
import random
from itertools import product

import pandas as pd

from setlexsem.constants import PATH_RESULTS_ROOT, PATH_ROOT
from setlexsem.experiment.experiment import run_experiment
from setlexsem.experiment.lmapi import LMClass
from setlexsem.generate import load_generated_data
from setlexsem.generate.prompt import PromptConfig
from setlexsem.generate.sample import (
    BasicNumberSampler,
    BasicWordSampler,
    DeceptiveWordSampler,
    DecileWordSampler,
    OverlapSampler,
)
from setlexsem.utils import get_study_paths, read_config


def replace_none(list_in):
    return [None if x == "None" else x for x in list_in]


# define argparser
def parse_args():
    parser = argparse.ArgumentParser()
    # add account number
    parser.add_argument("--account_number", type=str, help="account number")
    parser.add_argument(
        "--config_file",
        type=str,
        default=os.path.join(PATH_ROOT, "config.yaml"),
        help="save files to disk",
    )
    parser.add_argument(
        "--save_files",
        type=int,
        help="save files to disk",
    )
    parser.add_argument(
        "--load_last_run",
        type=int,
        default=1,
        help="load last run",
    )
    parser.add_argument(
        "--debug_model_no_lm_call",
        type=int,
        default=0,
        help="debug model without calling language model",
    )
    args = parser.parse_args()
    return args


# init
if __name__ == "__main__":
    # parse args
    args = parse_args()
    ACCOUNT_NUMBER = args.account_number
    CONFIG_FILE = args.config_file
    DEBUG_MODEL_NO_LM_CALL = True if args.debug_model_no_lm_call else False
    SAVE_FILES = True if args.save_files else False
    LOAD_LAST_RUN = True if args.load_last_run else False

    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(level=logging.INFO)

    # TODO: Add this to Config
    SWAP_STATUS = False

    # Read Config File and Assign Variables
    config = read_config(CONFIG_FILE)

    # Experiment config
    PATH_RESULTS = PATH_RESULTS_ROOT
    STUDY_NAME = config["STUDY_NAME"]
    N_RUN = config["N_RUN"]
    LOAD_GENERATED_DATA = config["LOAD_GENERATED_DATA"]
    RANDOM_SEED_VAL = config["RANDOM_SEED_VAL"]
    OP_LIST = config["OP_LIST"]
    MODEL_NAME = config["MODEL_NAME"]

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

    # create the LLM class
    LM = LMClass(MODEL_NAME, account_number=ACCOUNT_NUMBER)

    # go through hyperparameters and run the experiment
    counter_exp = 1
    for hp in make_hps():
        for hp_prompt in make_hps_prompt():
            LOGGER.info(
                f"-------- EXPERIMENT #{counter_exp} out of {n_experiments}"
            )
            # initilize the last run check
            last_run_check = False
            df_last_run = pd.DataFrame()
            path_study, path_results = get_study_paths(
                hp,
                hp_prompt,
                random_seed=RANDOM_SEED_VAL,
                study_name=STUDY_NAME,
                path_root=PATH_RESULTS,
            )
            if os.path.exists(path_results):
                if LOAD_LAST_RUN:
                    last_run_check = True
                    # LOGGER.info(f"Loading Last Run: {path_results}")
                    df_last_run = pd.read_csv(path_results)
                    last_run_count = len(df_last_run)
                    N_RUN_LEFT = N_RUN - last_run_count
                    if N_RUN <= last_run_count:
                        LOGGER.warning(
                            "--> Skipping, model is saved for all n-runs"
                        )
                        counter_exp += 1
                        continue

                else:
                    LOGGER.error(f"--> Skipping, file exists: {path_results}")
                    counter_exp += 1
                    continue
            else:
                N_RUN_LEFT = N_RUN

            if N_RUN_LEFT != N_RUN:
                LOGGER.info(f"Changed number of runs to {N_RUN_LEFT}")

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

            if LOAD_GENERATED_DATA:
                # NOTE: k-shot sampler has to be defined before loading data
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

            # generate samples from the sampler to reach the last run count
            if last_run_check:
                for i in range(last_run_count):
                    if LOAD_GENERATED_DATA:
                        A, B = next(sampler)
                    else:
                        A, B = sampler()

                    check_A = ast.literal_eval(df_last_run.iloc[i]["set_A"])
                    check_B = ast.literal_eval(df_last_run.iloc[i]["set_B"])
                    assert ast.literal_eval(A) == check_A, (
                        f"Run #{i} is incompatible with last run --> "
                        f"{A} is not {check_A}.\n\nCheck: {path_results}"
                    )
                    assert ast.literal_eval(B) == check_B, (
                        f"Run #{i} is incompatible with last run --> "
                        "{B} is not {check_B}\n\nCheck: {path_results}"
                    )

            # Run Experiment
            try:
                results, exp_logs = run_experiment(
                    LM,
                    sampler,
                    prompt_config,
                    num_runs=N_RUN_LEFT,
                    debug_no_lm=DEBUG_MODEL_NO_LM_CALL,
                )
            except:
                LOGGER.error("------> Error: Skipping this experiment")
                counter_exp += 1
                continue

            df_results = pd.DataFrame(exp_logs)
            # concatenate with last run data (if exists, if not, it's empty)
            df_results = pd.concat([df_last_run, df_results], axis=0)

            # Save Results
            df_op = df_results.reset_index(drop=True).copy()

            # save df_results
            if SAVE_FILES:
                if not os.path.exists(path_study):
                    os.makedirs(path_study)

                # save results
                df_op.to_csv(path_results, index=False)
                LOGGER.info(f"--> file saved at {path_results}")

            counter_exp += 1

    LOGGER.info("Done!")
