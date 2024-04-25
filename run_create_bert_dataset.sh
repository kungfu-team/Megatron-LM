#!/bin/bash
set -e

torchrun create_bert_dataset.py --seq-length 1024
