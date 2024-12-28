from itertools import product
from typing import Dict, List, Union
from unittest.mock import Mock

import pytest

from setlexsem.generate.generate_prompts import (
    get_prompt_config,
    make_hps_prompt,
    replace_none,
)
from setlexsem.generate.prompt import PromptConfig


def test_replace_none_empty_list():
    # Test with empty list
    assert replace_none([]) == []


def test_replace_none_no_replacement_needed():
    # Test with list containing no "None" strings
    assert replace_none(["a", "b", "c"]) == ["a", "b", "c"]


def test_replace_none_all_none():
    # Test with list containing all "None" strings
    assert replace_none(["None", "None", "None"]) == [None, None, None]


def test_replace_none_mixed_types():
    # Test with list containing different types
    assert replace_none([1, "None", 3.14, "None", True]) == [
        1,
        None,
        3.14,
        None,
        True,
    ]


def test_replace_none_case_sensitive():
    # Test that only exact "None" string is replaced
    assert replace_none(["none", "NONE", "None", "NoNe"]) == [
        "none",
        "NONE",
        None,
        "NoNe",
    ]


def test_replace_none_with_actual_none():
    # Test with list containing actual None values
    assert replace_none([None, "None", None]) == [None, None, None]


def test_replace_none_with_whitespace():
    # Test that whitespace is not affected
    assert replace_none([" None", "None ", " None "]) == [
        " None",
        "None ",
        " None ",
    ]


# Test fixtures
@pytest.fixture
def sample_config():
    return {
        "op_list": ["op1", "op2"],
        "k_shot": [1, 2],
        "prompt_type": ["type1"],
        "prompt_approach": ["approach1"],
        "is_fix_shot": [True],
    }


def test_make_hps_prompt_with_single_values():
    # Test with single values for each parameter
    result = list(
        make_hps_prompt(
            op_list="op1",
            k_shot=1,
            prompt_type="type1",
            prompt_approach="approach1",
            is_fix_shot=True,
        )
    )

    assert len(result) == 1
    assert result[0] == {
        "op_list": "op1",
        "k_shot": 1,
        "prompt_type": "type1",
        "prompt_approach": "approach1",
        "is_fix_shot": True,
    }


def test_make_hps_prompt_with_lists():
    # Test with lists for some parameters
    result = list(
        make_hps_prompt(
            op_list=["op1", "op2"],
            k_shot=[1, 2],
            prompt_type="type1",
            prompt_approach="approach1",
            is_fix_shot=True,
        )
    )

    assert len(result) == 4  # 2 op_list values × 2 k_shot values
    assert {
        "op_list": "op1",
        "k_shot": 1,
        "prompt_type": "type1",
        "prompt_approach": "approach1",
        "is_fix_shot": True,
    } in result
    assert {
        "op_list": "op1",
        "k_shot": 2,
        "prompt_type": "type1",
        "prompt_approach": "approach1",
        "is_fix_shot": True,
    } in result
    assert {
        "op_list": "op2",
        "k_shot": 1,
        "prompt_type": "type1",
        "prompt_approach": "approach1",
        "is_fix_shot": True,
    } in result
    assert {
        "op_list": "op2",
        "k_shot": 2,
        "prompt_type": "type1",
        "prompt_approach": "approach1",
        "is_fix_shot": True,
    } in result


def test_make_hps_prompt_with_config(sample_config):
    # Test using config dictionary
    result = list(make_hps_prompt(config=sample_config))

    assert len(result) == 4  # 2 op_list values × 2 k_shot values
    assert all(isinstance(r, dict) for r in result)
    assert all(len(r) == 5 for r in result)  # Each dict should have 5 keys


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

    # Wrap each parameter in a list if it isn't already, to enable Cartesian product
    param_grid = {
        "op_list": op_list if isinstance(op_list, list) else [op_list],
        "k_shot": k_shot if isinstance(k_shot, list) else [k_shot],
        "prompt_type": prompt_type
        if isinstance(prompt_type, list)
        else [prompt_type],
        "prompt_approach": prompt_approach
        if isinstance(prompt_approach, list)
        else [prompt_approach],
        "is_fix_shot": is_fix_shot
        if isinstance(is_fix_shot, list)
        else [is_fix_shot],
    }

    # Generate combinations of all parameters as dictionaries
    keys, values = zip(*param_grid.items())
    return (dict(zip(keys, v)) for v in product(*values))


def test_make_hps_prompt_empty_config():
    # Test with empty config and no parameters
    result = list(make_hps_prompt(config={}))  # Pass empty dict explicitly
    assert len(result) == 1
    expected = {
        "op_list": None,
        "k_shot": None,
        "prompt_type": None,
        "prompt_approach": None,
        "is_fix_shot": None,
    }
    assert result[0] == expected


def test_make_hps_prompt_mixed_config():
    # Test with both config and direct parameters
    # Note: When config is provided, it completely overrides direct parameters
    config = {
        "op_list": ["config_op"],
        "k_shot": [3],
    }
    result = list(
        make_hps_prompt(
            op_list="direct_op", k_shot=1, prompt_type="type1", config=config
        )
    )

    assert len(result) == 1
    expected = {
        "op_list": "config_op",
        "k_shot": 3,
        "prompt_type": None,  # Should be None since not in config
        "prompt_approach": None,
        "is_fix_shot": None,
    }
    assert result[0] == expected


# Additional test to verify config takes complete precedence
def test_make_hps_prompt_config_precedence():
    config = {
        "op_list": ["config_op"],
        "k_shot": [3],
        "prompt_type": ["config_type"],
    }
    result = list(
        make_hps_prompt(
            op_list="direct_op",
            k_shot=1,
            prompt_type="direct_type",
            config=config,
        )
    )

    assert len(result) == 1
    assert result[0]["op_list"] == "config_op"
    assert result[0]["k_shot"] == 3
    assert result[0]["prompt_type"] == "config_type"


def test_make_hps_prompt_none_values():
    # Test handling of None values in lists
    result = list(make_hps_prompt(op_list=[None, "op1"], k_shot=None))

    assert len(result) == 2
    assert any(r["op_list"] is None for r in result)
    assert all(r["k_shot"] is None for r in result)


def test_make_hps_prompt_type_consistency():
    # Test that the function preserves input types
    result = list(
        make_hps_prompt(
            op_list=["op1"],
            k_shot=[1],  # integer
            is_fix_shot=[True],  # boolean
        )
    )

    assert len(result) == 1
    assert isinstance(result[0]["k_shot"], int)
    assert isinstance(result[0]["is_fix_shot"], bool)


def test_make_hps_prompt_empty_lists():
    # Test handling of empty lists
    result = list(make_hps_prompt(op_list=[], k_shot=[1]))

    assert len(result) == 0  # Should generate no combinations with empty list


def test_get_prompt_config():
    # Create a mock Sampler instance
    mock_sampler = Mock()
    mock_sampler.get_members_type.return_value = "test_type"

    # Create a sample input dictionary
    test_prompt_config = {
        "op_list": "test_operation",
        "k_shot": 5,
        "prompt_type": "test_type",
        "prompt_approach": "test_approach",
        "is_fix_shot": True,
    }

    # Call the function
    result = get_prompt_config(test_prompt_config, mock_sampler)

    # Assert that the returned object is an instance of PromptConfig
    assert isinstance(result, PromptConfig)

    # Assert that all attributes are set correctly
    assert result.operation == "test_operation"
    assert result.k_shot == 5
    assert result.type == "test_type"
    assert result.approach == "test_approach"
    assert result.is_fixed_shots == True
    assert result.sampler == mock_sampler
    assert result.item_type == "test_type"


def test_get_prompt_config_with_different_values():
    # Test with different values
    mock_sampler = Mock()
    mock_sampler.get_members_type.return_value = "another_type"

    test_prompt_config = {
        "op_list": "another_operation",
        "k_shot": 3,
        "prompt_type": "another_type",
        "prompt_approach": "another_approach",
        "is_fix_shot": False,
    }

    result = get_prompt_config(test_prompt_config, mock_sampler)

    assert isinstance(result, PromptConfig)
    assert result.operation == "another_operation"
    assert result.k_shot == 3
    assert result.type == "another_type"
    assert result.approach == "another_approach"
    assert result.is_fixed_shots == False
    assert result.sampler == mock_sampler
    assert result.item_type == "another_type"


def test_get_prompt_config_invalid_input():
    mock_sampler = Mock()

    # Test with missing keys
    invalid_config = {
        "op_list": "test_operation",
        "k_shot": 5
        # missing other required keys
    }

    with pytest.raises(KeyError):
        get_prompt_config(invalid_config, mock_sampler)
