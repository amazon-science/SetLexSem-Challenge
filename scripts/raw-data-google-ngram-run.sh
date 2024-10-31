#!/bin/bash
#
# Download and run a script that automatically downloads the Google Books Ngram
# Viewer corpus of tokens and their counts.

# Exit if any command fails (i.e. returns a non-zero).
set -e

if [ ! -e raw-data-google-ngram ]
then
    git clone https://github.com/stressosaurus/raw-data-google-ngram.git
fi

# Unigrams, as we only care about single words.
ngram=1
# Language (see section 3 of https://stressosaurus.github.io/raw-data-google-ngram/
# for other language codes).
language=eng
# Start year of data in corpus.
start_year=2008
# End year of data in corpus.
end_year=2008
# Eliminate words that occur fewer than this many times.
min_word_count=1
# Eliminate words that occur in fewer than this many books.
min_book_count=1

# Download (verbosely with -x).
cd raw-data-google-ngram && sh -x ./downloadAndFilter.ngram.sh \
    $n $l \
    $start_year $end_year \
    $min_word_count $min_book_count
