import json
from argparse import ArgumentParser

from nltk.corpus import words

from setlexsem.prepare.ngrams import (
    get_counts_dict_from_google_books,
    make_percentiles,
)


def get_parser():
    parser = ArgumentParser(
        description=(
            "Make a JSON file with English words grouped into their k-th "
            "percentiles, as determined by their rank frequency. The words "
            "are from the NLTK english dictionary and the frequencies are "
            "from the Google Books Ngram corpus."
        )
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        required=True,
        help=(
            "The k-th percentile to use (1 is percentile, 10 is decile, etc.)"
        )
    )
    parser.add_argument(
        "--google-ngrams-path",
        type=str,
        required=True,
        help="Path to Google Books Ngram file (e.g. frequencies.txt)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to which to write the percentiles dictionary.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if not 1 <= args.k <= 50:
        raise ValueError("-k argument should be >= 1 and <= 50")

    words = set(w.lower() for w in words.words())
    words_to_counts = get_counts_dict_from_google_books(
        words, args.google_ngrams_path
    )

    percentiles = make_percentiles(words_to_counts, args.k)

    with open(args.output_path, "wt") as fh:
        json.dump(percentiles, fh)
