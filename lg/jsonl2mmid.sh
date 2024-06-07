#!/bin/sh
set -e

flags() {
    echo --vocab-file "/data/megatron-lm/gpt-2/gpt2-vocab.json"
    echo --merge-file "/data/megatron-lm/gpt-2/gpt2-merges.txt"
    echo --dataset-impl mmap
    echo --tokenizer-type GPT2BPETokenizer
    echo --workers 1
    # echo --chunk-size 1024
}

jsonl2mmid() {
    python3 ./jsonl2mmid.py --input "$1" --output-prefix "$2" $(flags)
}

# python3 -m pip install nltk

jsonl2mmid /data/enwiki/json/tiny_wiki.jsonl /data/lg/enwiki
