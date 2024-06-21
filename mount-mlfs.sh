#!/bin/sh
set -e

sudo systemctl restart mlfs

idx=/data/megatron-lm/gpt-2/enwiki/npzs_seq1024_new/indices.txt
# idx=/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt

mlfs mount \
    -index-url $idx \
    -job tenplex-samples \
    -global-batch-size 128 \
    -no-shuffle
