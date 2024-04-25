#!/bin/bash

set -e

i=0
for f in /data/the-pile/*.jsonl; do
    python tools/preprocess_data.py \
            --input $f \
            --output-prefix "/data/megatron-lm/gpt-2/the-pile/$i" \
            --vocab "/data/megatron-lm/gpt-2/gpt2-vocab.json" \
            --merge-file "/data/megatron-lm/gpt-2/gpt2-merges.txt" \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --workers 1 \
            --chunk-size 1024
    i=$((i+1))
done
