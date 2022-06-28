#!/bin/bash

set -e

python tools/preprocess_data.py \
	--input /data/enwiki/json/wiki.json \
	--output-prefix bert \
	--vocab /data/megatron-lm/bert/bert-large-uncased-vocab.txt \
	--dataset-impl mmap \
	--tokenizer-type BertWordPieceLowerCase \
	--split-sentences \
        --workers 32 \
        --chunk-size 1024
