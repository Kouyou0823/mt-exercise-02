#!/bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data
mkdir -p $data

tools=$base/tools

# link default training data for easier access
mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set: Pride and Prejudice
mkdir -p $data/pride_prejudice/raw

wget -O $data/pride_prejudice/raw/pride_prejudice.txt https://www.gutenberg.org/cache/epub/1342/pg1342.txt

# preprocess slightly
cat $data/pride_prejudice/raw/pride_prejudice.txt | python $base/scripts/preprocess_raw.py > $data/pride_prejudice/raw/pride_prejudice.cleaned.txt

# tokenize, fix vocabulary upper bound
cat $data/pride_prejudice/raw/pride_prejudice.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" > \
    $data/pride_prejudice/raw/pride_prejudice.preprocessed.txt

# split into train, valid and test (80% - 10% - 10%)
total_lines=$(wc -l < $data/pride_prejudice/raw/pride_prejudice.preprocessed.txt)
train_lines=$((total_lines * 80 / 100))
valid_lines=$((total_lines * 10 / 100))
test_lines=$((total_lines - train_lines - valid_lines))

head -n $train_lines $data/pride_prejudice/raw/pride_prejudice.preprocessed.txt > $data/pride_prejudice/train.txt
head -n $((train_lines + valid_lines)) $data/pride_prejudice/raw/pride_prejudice.preprocessed.txt | tail -n $valid_lines > $data/pride_prejudice/valid.txt
tail -n $test_lines $data/pride_prejudice/raw/pride_prejudice.preprocessed.txt > $data/pride_prejudice/test.txt
