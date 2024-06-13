#!/bin/bash
set -e

mlfs mount -index-url "/data/megatron-lm/gpt-2/enwiki/npzs_seq1024_new/indices.txt" -job tenplex-samples -global-batch-size 128 -no-shuffle
docker run --rm -it -v /mnt/mlfs:/data/mlfs -v ./out:/data/out kungfu.azurecr.io/mw-megatron-lm-23.06-debug:latest /bin/bash
