#!/bin/sh

set -e

cd $(dirname $0)

export CUDA_DEVICE_MAX_CONNECTIONS=1

# CHECKPOINT_PATH=<Specify path>
# VOCAB_FILE=<Specify path to file>/bert-vocab.txt
# DATA_PATH=<Specify path and file prefix>_text_sentence

data_flags() {
    echo --data-path $DATA_PATH
    echo --vocab-file $VOCAB_FILE
    echo --data-impl mmap
    echo --split 949,50,1
}

output_flags() {
    echo --log-interval 100
    echo --save-interval 10000
    echo --eval-interval 1000
    echo --eval-iters 10
}

flags() {
    ./bert-args.sh
    data_flags
    output_flags
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
}

torchrun pretrain_bert.py $(flags)

echo "done $0"
