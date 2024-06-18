#!/bin/sh
set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH=$HOME/var/ckpt/gpt
VOCAB_FILE=/data/megatron-lm/gpt-2/gpt2-vocab.json
MERGE_FILE=/data/megatron-lm/gpt-2/gpt2-merges.txt
# DATA_PATH=/data/megatron-lm/gpt-2/enwiki/tiny_text_document
DATA_PATH=/data/megatron-lm/gpt-2/enwiki/data_text_document

flags() {
    echo --num-layers 24
    echo --hidden-size 1024
    echo --num-attention-heads 16
    echo --seq-length 1024
    echo --max-position-embeddings 1024
    echo --micro-batch-size 4
    echo --global-batch-size 8
    echo --lr 0.00015
    echo --train-iters 500000
    echo --lr-decay-iters 320000
    echo --lr-decay-style cosine
    echo --min-lr 1.0e-5
    echo --weight-decay 1e-2
    echo --lr-warmup-fraction .01
    echo --clip-grad 1.0
    echo --fp16

    echo --data-path $DATA_PATH
    echo --vocab-file $VOCAB_FILE
    echo --merge-file $MERGE_FILE
    echo --data-impl mmap
    echo --split 949,50,1
}

torchrun ./pretrain_gpt.py $(flags)
