import glob
import gzip
import logging
import os
import shutil
import tempfile
import urllib.request

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


def download_files(language, output_dir):
    domain = "http://storage.googleapis.com"
    path = "books/ngrams/books"
    baseurl = f"{domain}/{path}/googlebooks-{language}-all-1gram-20120701"

    for char in range(ord("a"), ord("z") + 1):
        url = f"{baseurl}-{chr(char)}.gz"
        filename = os.path.join(output_dir, os.path.basename(url))
        urllib.request.urlretrieve(url, filename)


def filter_file(
    gzip_path, start_year, end_year, min_word_count, min_book_count
):
    output_path = gzip_path.replace(".gz", ".filtered")
    with gzip.open(gzip_path, "rt") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            parts = line.split()
            if len(parts) >= 4:
                year, word_count, book_count = map(int, parts[1:4])
                if (
                    start_year <= year <= end_year
                    and word_count >= min_word_count
                    and book_count >= min_book_count
                ):
                    f_out.write(line)


def make_term_frequencies_file(output_path="frequencies.txt"):
    language = "eng"
    start_year = 2008
    end_year = 2008
    min_word_count = 1
    min_book_count = 1

    with tempfile.TemporaryDirectory() as tempdir:
        LOGGER.info(
            f"Downloading Google Books Ngram corpus files to {tempdir}..."
        )
        download_files(language, tempdir)
        LOGGER.info("... done")

        LOGGER.info("Filtering terms in downloaded files...")
        for gzip_path in glob.glob(os.path.join(tempdir, "*.gz")):
            LOGGER.info(f"    => {gzip_path}")
            filter_file(
                gzip_path,
                start_year,
                end_year,
                min_word_count,
                min_book_count,
            )
        LOGGER.info("... done")

        LOGGER.info("Writing frequencies.txt to current directory...")
        with open("frequencies.txt", "wb") as output_path:
            for filtered_file in glob.glob(
                os.path.join(tempdir, "*.filtered")
            ):
                with open(filtered_file, "rb") as infile:
                    shutil.copyfileobj(infile, output_path)
        LOGGER.info("... done")

    LOGGER.info(f"Temporary directory {tempdir} has been deleted.")


def get_term_frequencies(term_frequency_output_path="frequencies.txt"):
    make_term_frequencies_file(output_path=term_frequency_output_path)
    with open(term_frequency_output_path, "rt") as fh:
        # The frequencies.txt file is about 140MB in memory.
        term_frequencies = [line.strip() for line in fh.readlines()]
        return term_frequencies


if __name__ == "__main__":
    term_frequencies = get_term_frequencies()
