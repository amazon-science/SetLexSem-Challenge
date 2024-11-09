import random
import argparse

from setlexsem.generate.sample import get_clean_hyponyms


def get_parser():
    parser = argparse.ArgumentParser(
        description="Make a JSON file containing sets of hyponyms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling."
    )
    parser.add_argument(
        "--output-path",
        required=True,
        default="data/hyponyms.json",
        help="Path to which to write the hyponyms."
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    random_state = random.Random(args.seed)
    get_clean_hyponyms(
        random_state, save_json=1, filename=args.output_path
    )
