#!/bin/bash

scripts/download-google-ngrams.sh
mkdir -p data
python scripts/make_percentiles.py \
    --k 10 \
    --google-ngrams-path frequencies.txt \
    --output-path data/deciles.json
rm -i frequencies.txt
