import pytest

from setlexsem.experiment.experiment import run_experiment
from setlexsem.experiment.lmapi import LMClass
from setlexsem.generate.prompt import (
    OPS,
    PromptConfig,
    make_instruction_generator,
)
from setlexsem.generate.sample import BasicNumberSampler


@pytest.fixture
def n():
    return 100


@pytest.fixture
def m():
    return 3


@pytest.fixture
def num_runs():
    return 1


def get_account_number():
    return input("Enter account number: ")


ACCOUNT_NUMBER = get_account_number()

# define LMClass to test the functions
LM = LMClass(
    model_name="anthropic.claude-instant-v1", account_number=ACCOUNT_NUMBER
)


def assert_results_are_ok(results, ops, errors):
    try:
        assert isinstance(results, dict)
        if len(results) < len(OPS):
            assert len(errors)
        for op, num_correct in results.items():
            assert isinstance(op, str)
            assert op in ops
            assert isinstance(num_correct, int)
            assert num_correct >= 0
    except AssertionError as e:
        print(f"results {results}")
        print(f"ops {ops}")
        print(f"errors {errors}")
        raise e


def test_zero_shot_with_formal_language(n, m, num_runs):
    """
    Zero-shot prompt with formal language
    """
    number_sampler = BasicNumberSampler(n=n, m=m)
    k_shot = 0
    results = {}
    for op in OPS:
        prompt_config = PromptConfig(
            operation=op,
            k_shot=k_shot,
            type="formal_language",
            approach="baseline",
            sampler=number_sampler,
            is_fixed_shots=True,
        )

        results[op], errors = run_experiment(
            LM,
            sampler=number_sampler,
            prompt_config=prompt_config,
            num_runs=num_runs,
        )

    assert_results_are_ok(results, OPS, errors)


def test_one_shot_with_formal_language(n, m, num_runs):
    """
    One-shot prompt using formal language
    """
    number_sampler = BasicNumberSampler(n=n, m=m)
    k_shot = 1
    results = {}
    for op in OPS:
        prompt_config = PromptConfig(
            operation=op,
            k_shot=k_shot,
            type="formal_language",
            approach="baseline",
            sampler=number_sampler,
            is_fixed_shots=True,
        )

        results[op], errors = run_experiment(
            LM,
            sampler=number_sampler,
            prompt_config=prompt_config,
            num_runs=num_runs,
        )

    assert_results_are_ok(results, OPS, errors)


def test_five_shot_with_formal_language(n, m, num_runs):
    """
    Five-shot prompting using formal language
    """
    number_sampler = BasicNumberSampler(n=n, m=m)

    k_shot = 5
    results = {}
    for op in OPS:
        prompt_config = PromptConfig(
            operation=op,
            k_shot=k_shot,
            type="formal_language",
            approach="baseline",
            sampler=number_sampler,
            is_fixed_shots=True,
        )

        results[op], errors = run_experiment(
            LM,
            sampler=number_sampler,
            prompt_config=prompt_config,
            num_runs=num_runs,
        )

    assert_results_are_ok(results, OPS, errors)
