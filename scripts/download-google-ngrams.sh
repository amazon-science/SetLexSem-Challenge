#!/bin/bash

# Download and preprocess (English) unigrams from
# http://storage.googleapis.com/books/ngrams/books/datasetsv2.html

function download() {
	language=$1
	output_dir=$2
	domain=http://storage.googleapis.com
	baseurl=$domain/books/ngrams/books/googlebooks-${language}-all-1gram-20120701

	for char in {a..z}
	do
	    wget -q --directory-prefix=$output_dir $baseurl-${char}.gz
	done
}

function filter() {
    gzip_path=$1
    start_year=$2
    end_year=$3
    min_word_count=$4
    min_book_count=$5
    output_path=${gzip_path/.gz/.filtered}
    gunzip -c $gzip_path |
        awk -v var="$start_year" '$2 >= var' |
        awk -v var="$end_year" '$2 <= var' |
		awk -v var="$min_word_count" '$3 >= var' |
		awk -v var="$min_book_count" '$4 >= var' > $output_path
}

language=eng
start_year=2008
end_year=2008
min_word_count=1
min_book_count=1

tempdir=$(mktemp -d)
echo "Downloading Google Books Ngram corpus files to $tempdir..."
download $language $tempdir
echo "... done"

echo "Filtering terms in downloaded files..."
for gzip_path in $tempdir/*.gz
do
    echo "    => $gzip_path"
	filter $gzip_path $start_year $end_year $min_word_count $min_book_count
done
echo "... done"

echo "Writing frequencies.txt to current directory..."
cat $tempdir/*.filtered > frequencies.txt
echo "... done"

echo "Deleting $tempdir..."
rm -fr $tempdir
echo "... done"
