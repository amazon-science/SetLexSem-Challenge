# coding: utf-8

import ast
import logging
from collections.abc import Iterable

from tqdm import tqdm

from setlexsem.experiment.lmapi import get_context_length, parse_lm_response
from setlexsem.generate.prompt import get_ground_truth, get_prompt, is_correct

# define the logger
logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=logging.WARNING)


def run_experiment(
    lm,
    sampler,
    prompt_config,
    num_runs=100,
    debug_no_lm=False,
):
    results = 0
    experiment_logs = []
    lm_model_owner = lm.get_model_owner()
    lm_model_name = lm.get_model_name()
    add_roles = False
    if "anthropic" in lm_model_name:
        if "claude-3" not in lm_model_name:
            add_roles = True

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
        if debug_no_lm:
            result = "set()"
        else:
            result = lm(prompt)

        dict_context_length = get_context_length(
            prompt_in=prompt,
            prompt_out=result,
            model_owner=lm_model_owner,
            model_name=lm_model_name,
        )
        ground_truth = get_ground_truth(prompt_config.operation, A, B)
        # log the conversation
        LOGGER.info(
            f"\n{prompt}\n"
            f"LM Response: {result}\n"
            f"GT Response: {ground_truth}"
        )
        try:
            # postprocess lm response
            result_obj = parse_lm_response(result)
            # compare with groundtruth
            ok = is_correct(ground_truth, result_obj)
            results += int(ok)
        except Exception as e:
            result_obj = {-1}  # did not follow guideline
            ok = False
            LOGGER.warning(
                f"op {prompt_config.operation} failed:\n"
                f"--> result {result}\n"
                f"------> exception {e}"
            )

        # log all experiments
        experiment_log = {
            "op_name": prompt_config.operation,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "result_obj": result_obj,
            "llm_vs_gt": ok,
            "set_A": A,
            "set_B": B,
            "context_length_in": dict_context_length["in"],
            "context_length_out": dict_context_length["out"],
            "log_context": prompt + result,
        }
        experiment_logs.append(experiment_log)

    return results, experiment_logs
