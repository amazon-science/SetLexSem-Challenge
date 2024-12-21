import pytest

from setlexsem.generate.sample import (  # Adjust import as necessary
    make_sampler_name_from_hps,
)


@pytest.mark.parametrize(
    "sampler_hps, expected",
    [
        (
            {
                "m_A": 5,
                "m_B": 6,
                "item_len": 3,
                "set_type": "decile",
                "decile_num": 3,
            },
            "MA-5_MB-6_L-3_Decile-3",
        ),
        (
            {
                "m_A": 5,
                "m_B": 6,
                "item_len": 2,
                "set_type": "words",
                "overlap_fraction": 0.5,
            },
            "MA-5_MB-6_L-2_O-0.5",
        ),
        (
            {
                "m_A": 4,
                "m_B": 4,
                "item_len": 4,
                "set_type": "words",
                "overlap_fraction": 0.75,
                "decile_num": 3,
            },
            "MA-4_MB-4_L-4_O-0.75_Decile-3",
        ),
        (
            {
                "n": 10,
                "m_A": 5,
                "m_B": 5,
                "item_len": None,
                "set_type": "numbers",
                "overlap_fraction": None,
            },
            "N-10_MA-5_MB-5_L-None",
        ),
        (
            {
                "n": 10,
                "m_A": 2,
                "m_B": 4,
                "item_len": 2,
                "set_type": "numbers",
            },
            "N-None_MA-2_MB-4_L-2",
        ),
        (
            {
                "m_A": 5,
                "m_B": 2,
                "item_len": 2,
                "set_type": "decile",
                "decile_num": 5,
            },
            "MA-5_MB-2_L-2_Decile-5",
        ),
    ],
)
def test_make_sampler_name_from_hps(sampler_hps, expected):
    assert make_sampler_name_from_hps(sampler_hps) == expected
