"""
Helper functions for partitioning words according to their frequencies. See
uses of these functions in `scripts/make_percentiles.py`.
"""

import math
from collections import defaultdict
from operator import itemgetter
from typing import Set

import numpy as np


def get_counts_dict_from_google_books(words: Set[str], ngram_path: str):
    """Get a dictionary mapping words to their frequencies."""
    words_to_counts = {}
    with open(ngram_path, "rt") as fh:
        for line in fh:
            token, year, count, num_books = line.split()
            if token in words:
                words_to_counts[token] = int(count)
    return words_to_counts


def remove_outliers(words_to_counts, to_remove=50):
    word_count_items = list(words_to_counts.items())
    # Sort in ascending order according to the second item (zero-indexed).
    word_count_items = sorted(word_count_items, key=itemgetter(1))
    # Remove outliers (here, just the 50 most frequent words).
    word_count_items = word_count_items[:-to_remove]
    return dict(word_count_items)


def normalize_counts(words_to_counts):
    # The counts are power-law distributed, so take their log.
    for word, count in words_to_counts.items():
        words_to_counts[word] = math.log(count)
    min_count = min([count for word, count in words_to_counts.items()])
    max_count = max([count for word, count in words_to_counts.items()])
    for word in words_to_counts.keys():
        # Shift and scale.
        words_to_counts[word] -= min_count
        words_to_counts[word] /= max_count


def make_percentiles(words_to_counts, k, remove_outliers=False):
    partition = defaultdict(list)

    if remove_outliers:
        # Remove the 50 most frequent words.
        words_to_counts = remove_outliers(words_to_counts)
    # Normalize counts to [0, 1].
    normalize_counts(words_to_counts)

    step_size = k / 100
    bins = np.arange(0, 1, step_size)
    assignments = np.digitize(
        list(words_to_counts.values()), bins, right=False
    )
    assignments = assignments.tolist()
    words_to_bins = dict(zip(words_to_counts.keys(), assignments))
    for word, assignment in words_to_bins.items():
        partition[assignment].append(word)
    return partition
