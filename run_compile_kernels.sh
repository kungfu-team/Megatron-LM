#!/bin/bash

docker run \
    --gpus all \
    --name tenplex-compile \
    -t kungfu.azurecr.io/mw-megatron-lm-data:latest \
    python compile_kernels.py
