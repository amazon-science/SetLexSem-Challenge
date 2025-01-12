import unittest
from unittest.mock import patch

import pytest

from setlexsem.utils import (
    create_filename,
    create_param_format,
    extract_values,
    read_config,
)


class TestReadConfig(unittest.TestCase):
    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_read_config(self, mock_safe_load, mock_open):
        # Set up the mock data
        mock_config = {
            "STUDY_NAME": "Test",
            "MODEL_NAME": "anthropic.claude-instant-v1",
            "N_RUN": 5,
            "LOAD_GENERATED_DATA": True,
            "RANDOM_SEED_VAL": 292,
            "OP_LIST": ["union", "intersection"],
            "SET_TYPES": ["numbers"],
            "N": [10],
            "M_A": [4],
            "M_B": [4],
            "ITEM_LEN": [3],
            "OVERLAP_FRACTION": [0],
            "DECILE_NUM": [-1],
            "K_SHOT": [4],
            "PROMPT_TYPE": ["formal_language"],
            "PROMPT_APPROACH": ["baseline"],
            "IS_FIX_SHOT": [True],
        }
        mock_safe_load.return_value = mock_config

        # Call the function and assert the output
        config_values = read_config()
        self.assertEqual(config_values["STUDY_NAME"], "Test")
        self.assertEqual(
            config_values["MODEL_NAME"], "anthropic.claude-instant-v1"
        )
        self.assertEqual(config_values["N_RUN"], 5)
        self.assertEqual(config_values["LOAD_GENERATED_DATA"], True)
        self.assertEqual(config_values["RANDOM_SEED_VAL"], 292)
        self.assertEqual(config_values["OP_LIST"], ["union", "intersection"])
        self.assertEqual(config_values["SET_TYPES"], ["numbers"])
        self.assertEqual(config_values["N"], [10])
        self.assertEqual(config_values["M_A"], [4])
        self.assertEqual(config_values["M_B"], [4])
        self.assertEqual(config_values["ITEM_LEN"], [3])
        self.assertEqual(config_values["K_SHOT"], [4])
        self.assertEqual(config_values["PROMPT_TYPE"], ["formal_language"])
        self.assertEqual(config_values["PROMPT_APPROACH"], ["baseline"])
        self.assertEqual(config_values["IS_FIX_SHOT"], [True])


@pytest.mark.parametrize(
    "sampler_name, random_seed_value, expected",
    [
        ("SamplerA", 42, "SamplerA_S-42"),
        ("TestSampler", 123, "TestSampler_S-123"),
        ("SamplerX", 999, "SamplerX_S-999"),
    ],
)
def test_create_param_format(sampler_name, random_seed_value, expected):
    assert create_param_format(sampler_name, random_seed_value) == expected


@pytest.mark.parametrize(
    "prompt_type, sampler_name, k_shot, random_seed_value, expected",
    [
        ("TypeA", "SamplerA", 5, 42, "TypeA_K-5_SamplerA_S-42.csv"),
        ("TypeB", "TestSampler", 10, 123, "TypeB_K-10_TestSampler_S-123.csv"),
        ("TypeC", "SamplerX", 1, 999, "TypeC_K-1_SamplerX_S-999.csv"),
    ],
)
def test_create_filename(
    prompt_type, sampler_name, k_shot, random_seed_value, expected
):
    assert (
        create_filename(prompt_type, sampler_name, k_shot, random_seed_value)
        == expected
    )


def test_basic_extraction():
    filename = "K-10_N-100_M-5_MA-3_MB-7_L-0.5_S"
    result = extract_values(filename)
    assert result == {
        "k_shots": "10",
        "max_value": "100",
        "n_items": "5",
        "n_items_in_A": "3",
        "n_items_in_B": "7",
        "item_len": "0.5",
    }


def test_missing_parameters():
    filename = "K-10_N-100_L-0.5_S"
    result = extract_values(filename)
    assert result == {
        "k_shots": "10",
        "max_value": "100",
        "item_len": "0.5",
    }


def test_decimal_parameter():
    filename = "K-10_N-100_MA-5_MB-3_L-0.5_O-0.75_S"
    result = extract_values(filename)
    assert result == {
        "k_shots": "10",
        "max_value": "100",
        "n_items_in_A": "5",
        "n_items_in_B": "3",
        "item_len": "0.5",
        "overlap": "0.75",
    }


def test_word_parameter():
    filename = "K-10_N-100_M-5_L-0.5_DeceptiveWords_S"
    result = extract_values(filename)
    assert result == {
        "k_shots": "10",
        "max_value": "100",
        "n_items": "5",
        "item_len": "0.5",
    }


def test_custom_config_check():
    filename = "K-10_N-100_MA-5_L-0.5_S"
    result = extract_values(filename)
    assert result == {
        "k_shots": "10",
        "max_value": "100",
        "n_items_in_A": "5",
        "item_len": "0.5",
    }


def test_complex_parameter_values():
    filename = "K-10.5_N-1e6_M-5_L-0.5-0.7_DeceptiveWords_Swapped-1_S"
    result = extract_values(filename)
    assert result == {
        "k_shots": "10.5",
        "max_value": "1e6",
        "n_items": "5",
        "item_len": "0.5-0.7",
        "swapped": "1",
    }


def test_empty_filename():
    result = extract_values("")
    assert result == {}


def test_invalid_filename():
    filename = "InvalidFilename"
    result = extract_values(filename)
    assert result == {}
