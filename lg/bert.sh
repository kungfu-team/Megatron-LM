#!/bin/sh

set -e

cd $(dirname $0)

. ./_x

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=$HOME/var/ckpt/bert
VOCAB_FILE=/data/megatron-lm/bert/bert-large-uncased-vocab.txt
# DATA_PATH=/data/megatron-lm/bert/enwiki/bert_text_sentence
DATA_PATH=/data/megatron-lm/gpt-2/enwiki/tiny_text_document

bert_flags() {
    echo --num-layers 24
    echo --hidden-size 1024
    echo --num-attention-heads 16
    echo --seq-length 512
    echo --max-position-embeddings 512
    echo --micro-batch-size 4
    echo --global-batch-size 8
    echo --lr 0.0001
    # echo --train-iters 2000000
    echo --train-iters 100
    # echo --train-iters 50
    echo --lr-decay-iters 990000
    echo --lr-decay-style linear
    echo --min-lr 0.00001
    echo --weight-decay 1e-2
    echo --lr-warmup-fraction .01
    echo --clip-grad 1.0
    echo --fp16
}

data_flags() {
    echo --data-path $DATA_PATH
    echo --vocab-file $VOCAB_FILE
    echo --data-impl mmap
    echo --split 949,50,1
}

output_flags() {
    echo --log-interval 10
    echo --save-interval 10000
    echo --eval-interval 1000
    echo --eval-iters 10
}

flags() {
    bert_flags
    data_flags
    output_flags
    echo --save $CHECKPOINT_PATH
    echo --load $CHECKPOINT_PATH
}

show_bert_model() {
    torchrun ./bert_model.py $(flags)
}

show_bert_data() {
    torchrun ./bert_data.py $(flags)
}

train_bert() {
    torchrun ./pretrain_bert.py $(flags)
    cp -v trace.txt /log
}

main() {
    measure show_bert_model
    measure show_bert_data
    # measure train_bert
}

measure main

echo "done $0"
