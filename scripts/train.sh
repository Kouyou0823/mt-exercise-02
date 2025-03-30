#!/bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
logs=$base/logs
data=$base/data
tools=$base/tools

mkdir -p $models
mkdir -p $logs

num_threads=4
device=""

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/pride_prejudice \
        --epochs 25 \
        --log-interval 50 \
        --emsize 250 \
        --nhid 250 \
        --dropout 0.8 \
        --tied \
        --save $models/model_d4.pt \
        --log-file $logs/log_d4.txt
)

echo "time taken:"
echo "$SECONDS seconds"
