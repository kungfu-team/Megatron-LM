#!/bin/bash

set -e

python tools/preprocess_data.py \
	--input "/data/enwiki/json/wiki.json" \
	--output-prefix "/data/megatron-lm/gpt-2/enwiki/data" \
	--vocab "/data/megatron-lm/gpt-2/gpt2-vocab.json" \
	--merge-file "/data/megatron-lm/gpt-2/gpt2-merges.txt" \
	--dataset-impl mmap \
	--tokenizer-type GPT2BPETokenizer \
        --workers 1 \
        --chunk-size 1024
