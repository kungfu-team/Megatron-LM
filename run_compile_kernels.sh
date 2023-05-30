#!/bin/bash
set -e

docker run \
    --gpus all \
    --name tenplex-compile \
    -t kungfu.azurecr.io/mw-megatron-lm:latest \
    python compile_kernels.py

docker commit tenplex-compile kungfu.azurecr.io/mw-megatron-lm-kernel:latest
docker push kungfu.azurecr.io/mw-megatron-lm-kernel:latest
