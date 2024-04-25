#!/bin/bash
set -e

torchrun create_gpt_dataset.py --seq-length 1024
