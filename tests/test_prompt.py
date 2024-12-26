import random

import pytest

from setlexsem.generate.sample import BasicNumberSampler, BasicWordSampler


@pytest.fixture
def n():
    return 100


@pytest.fixture
def m_A():
    return 4


@pytest.fixture
def m_B():
    return 8


@pytest.fixture
def item_len():
    return 3


@pytest.fixture
def random_state():
    return random.Random(17)


def test_smoke_sample_numbers(n, m_A, m_B, item_len, random_state):
    """Smoke test of sample_numbers."""
    # Test 1: Without Item Length
    number_sampler = BasicNumberSampler(
        n=n, m_A=m_A, m_B=m_B, random_state=random_state
    )
    A, B = number_sampler()
    assert A is not None
    assert B is not None
    assert len(A) == m_A, "A should be of length m"
    assert len(B) == m_B, "B should be of length m"
    assert max(A) <= n, "A should be <= n"
    assert max(B) <= n, "B should be <= n"
    # Test 2: With Item Length
    number_sampler = BasicNumberSampler(
        n=n, m_A=m_A, m_B=m_B, item_len=item_len, random_state=random_state
    )
    A, B = number_sampler()
    for a in A:
        assert (
            len(str(a)) == item_len
        ), f"member {a} in A is not of length {item_len}"
    for b in B:
        assert (
            len(str(b)) == item_len
        ), f"member {b} in B is not of length {item_len}"


def test_smoke_sample_words(m_A, m_B, item_len, random_state):
    """Smoke test of sample_words."""
    # Test 1: Without Item Length
    word_sampler = BasicWordSampler(
        m_A=m_A, m_B=m_B, random_state=random_state
    )
    A, B = word_sampler()
    assert A is not None
    assert B is not None
    assert len(A) == m_A, "A should be of length m"
    assert len(B) == m_B, "B should be of length m"
    # Test 2: With Item Length
    word_sampler = BasicWordSampler(
        m_A=m_A, m_B=m_B, item_len=item_len, random_state=random_state
    )
    A, B = word_sampler()
    for a in A:
        assert (
            len(a) == item_len
        ), f"member {a} in A is not of length {item_len}"
    for b in B:
        assert (
            len(b) == item_len
        ), f"member {b} in B is not of length {item_len}"
