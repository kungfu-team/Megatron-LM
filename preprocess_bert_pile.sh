#!/bin/bash

set -e

BASE_DIR="/data/megatron-lm/bert"

i=28
for f in /data/the-pile/*.jsonl; do
    python tools/preprocess_data.py \
            --input $f \
            --output-prefix "${BASE_DIR}/the-pile/bert_pile_${i}" \
            --vocab "${BASE_DIR}/bert-large-uncased-vocab.txt" \
            --dataset-impl mmap \
            --tokenizer-type BertWordPieceLowerCase \
            --split-sentences \
            --workers 1 \
            --chunk-size 1024
    i=$((i+1))
done
