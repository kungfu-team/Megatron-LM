#!/bin/bash
set -e

idx=/data/megatron-lm/gpt-2/enwiki/npzs_seq1024_new/indices.txt
# idx=/data/megatron-lm/gpt-2/enwiki/npzs_seq1024/indices.txt

mlfs mount \
    -index-url $idx \
    -job tenplex-samples \
    -global-batch-size 128 \
    -no-shuffle

# docker run \
#     --rm \
#     -it \
#     -v /mnt/mlfs:/data/mlfs \
#     -v ./out:/data/out \
#     -v /data/megatron-lm:/data/dataset \
#     kungfu.azurecr.io/mw-megatron-lm-23.06-debug:latest \
#     /bin/bash
