"""
Make a JSON file that partitions words into groups -- specifically,
into their k-th percentiles, as determined by their rank frequency.

The frequencies are obtained from the Google Book NGram corpus using:

    https://stressosaurus.github.io/raw-data-google-ngram/

Run the following commands to complete the task:
    ```
    scripts/raw-data-google-ngram-run.sh
    scripts/partition_words.py --args ...
    python scripts/partition_words.py \
        --k 10 \
        --google-ngram-path PATH/googlebooks-eng-all-1gram-20120701.filtered \
        --output-path deciles.json
    ```
"""

import json
from argparse import ArgumentParser

from nltk.corpus import words

from setlexsem.prepare.ngrams import (
    get_counts_dict_from_google_books,
    partition_words,
)


def get_parser():
    parser = ArgumentParser(
        description=(
            "Make a JSON file that partitions words into groups -- "
            "specifically, into their k-th percentiles, as determined by "
            "their rank frequency. The words are from the NLTK english "
            "dictionary and the frequencies are from the Google Books Ngram "
            "Viewer corpus."
        )
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        required=True,
        help="The k-th percentile to use (1 is percentile, 10 is decile, etc.)",
    )
    parser.add_argument(
        "--google-ngram-path",
        type=str,
        required=True,
        help=(
            "Path to Google Books Ngram file obtained using "
            "https://github.com/stressosaurus/raw-data-google-ngram. "
            "The file will be named something like "
            "googlebooks-eng-all-1gram-20120701.filtered."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to which to write the partitioned dictionary.",
    )
    return parser


"""
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


def partition_words(words_to_counts, k):
    partition = defaultdict(list)

    # Remove the 50 most frequent words.
    words_to_counts = remove_outliers(words_to_counts)
    # Normalize counts to [0, 1].
    normalize_counts(words_to_counts)

    step_size = args.k / 100
    bins = np.arange(0, 1, step_size)
    assignments = np.digitize(
        list(words_to_counts.values()),
        bins,
        right=False
    )
    assignments = assignments.tolist()
    words_to_bins = dict(zip(words_to_counts.keys(), assignments))
    for word, assignment in words_to_bins.items():
        partition[assignment].append(word)
    return partition
"""


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if not 1 <= args.k <= 50:
        raise ValueError("-k argument should be >= 1 and <= 50")

    words = set(w.lower() for w in words.words())
    words_to_counts = get_counts_dict_from_google_books(
        words, args.google_ngram_path
    )

    partition = partition_words(words_to_counts, args.k)

    with open(args.output_path, "wt") as fh:
        json.dump(partition, fh)
