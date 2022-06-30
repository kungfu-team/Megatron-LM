#!/bin/bash

set -e

# python tools/preprocess_data.py \
python tools/preprocess_data_dp_rank.py \
	--input "/data/enwiki/json/wiki.json" \
	--output-prefix "/data/megatron-lm/bert/bert" \
	--vocab "/data/megatron-lm/bert/bert-large-uncased-vocab.txt" \
	--dataset-impl mmap \
	--tokenizer-type BertWordPieceLowerCase \
	--split-sentences \
        --workers 1 \
        --chunk-size 1024
