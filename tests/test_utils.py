import unittest
from unittest.mock import patch

from setlexsem.utils import read_config


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
            "M": [4],
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
        self.assertEqual(config_values["M"], [4])
        self.assertEqual(config_values["ITEM_LEN"], [3])
        self.assertEqual(config_values["K_SHOT"], [4])
        self.assertEqual(config_values["PROMPT_TYPE"], ["formal_language"])
        self.assertEqual(config_values["PROMPT_APPROACH"], ["baseline"])
        self.assertEqual(config_values["IS_FIX_SHOT"], [True])
