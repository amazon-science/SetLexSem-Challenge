import ast
from typing import Any, Dict, Iterable, List, Tuple, Union
from unittest.mock import Mock

import pytest

from setlexsem.generate.generate_sets import (
    generate_set_pair,
    make_sets_from_sampler,
    parse_set_pair,
)


def test_parse_set_pair():
    # Test with valid inputs
    assert parse_set_pair("{1, 2, 3}", "{4, 5, 6}") == ({1, 2, 3}, {4, 5, 6})

    # Test with invalid inputs
    with pytest.raises(ValueError):
        parse_set_pair("{1, 2}", "invalid_set")


def test_generate_set_pair():
    # Test with an iterable
    iterable = [("{1, 2, 3}", "{4, 5, 6}")]
    assert generate_set_pair(iter(iterable)) == ({1, 2, 3}, {4, 5, 6})

    # Test with a callable
    def mock_sampler():
        return {1, 2, 3}, {4, 5, 6}

    assert generate_set_pair(mock_sampler) == ({1, 2, 3}, {4, 5, 6})

    # Test with an iterable that returns invalid input
    invalid_iterable = [("{1, 2}", "invalid_set")]
    assert generate_set_pair(iter(invalid_iterable)) == (None, None)

    # Test with a callable that raises an exception
    def mock_exception_sampler():
        raise ValueError("Mock exception")

    assert generate_set_pair(mock_exception_sampler) == (None, None)


def test_make_sets_from_sampler():
    # Test with a valid sampler and positive number of runs
    mock_sampler = Mock(side_effect=[({1, 2}, {3, 4}), ({5, 6}, {7, 8})])
    result = make_sets_from_sampler(mock_sampler, num_runs=2)
    assert result == [
        {"experiment_run": 0, "A": {1, 2}, "B": {3, 4}},
        {"experiment_run": 1, "A": {5, 6}, "B": {7, 8}},
    ]

    # Test with an invalid sampler
    mock_invalid_sampler = Mock(side_effect=ValueError("Mock exception"))
    result = make_sets_from_sampler(mock_invalid_sampler, num_runs=2)
    assert result == []

    # Test with non-positive number of runs
    result = make_sets_from_sampler(mock_sampler, num_runs=0)
    assert result == []
