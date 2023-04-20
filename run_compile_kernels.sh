#!/bin/bash

docker run \
    --gpus all \
    --name tenplex-compile \
    -t kungfu.azurecr.io/mw-megatron-lm-no-scheduler:latest \
    python compile_kernels.py
