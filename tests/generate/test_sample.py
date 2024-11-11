import pytest

from setlexsem.generate.sample import (  # Adjust import as necessary
    make_sampler_name_from_hps,
)


@pytest.mark.parametrize(
    "sampler_hps, expected",
    [
        (
            {
                "n": 10,
                "m": 5,
                "item_len": 3,
                "set_type": "decile",
                "decile_num": 3,
            },
            "N-None_M-5_L-3_Decile-3",
        ),
        (
            {
                "n": 10,
                "m": 5,
                "item_len": 2,
                "set_type": "words",
                "overlap_fraction": 0.5,
            },
            "N-None_M-5_L-2_O-0.5",
        ),
        (
            {
                "n": 10,
                "m": 4,
                "item_len": 4,
                "set_type": "words",
                "overlap_fraction": 0.75,
                "decile_num": 3,
            },
            "N-None_M-4_L-4_O-0.75_Decile-3",
        ),
        (
            {
                "n": 10,
                "m": 5,
                "item_len": None,
                "set_type": "numbers",
                "overlap_fraction": None,
            },
            "N-10_M-5_L-None",
        ),
        (
            {
                "n": 10,
                "m": 5,
                "item_len": 2,
                "set_type": "numbers",
                "decile_num": None,
            },
            "N-None_M-5_L-2",
        ),
        (
            {
                "n": 100,
                "m": 5,
                "item_len": 2,
                "set_type": "decile",
                "decile_num": 5,
            },
            "N-None_M-5_L-2_Decile-5",
        ),
    ],
)
def test_make_sampler_name_from_hps(sampler_hps, expected):
    assert make_sampler_name_from_hps(sampler_hps) == expected
