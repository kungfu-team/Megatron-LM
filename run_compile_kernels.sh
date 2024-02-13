#!/bin/bash
set -e

docker run \
    --gpus all \
    --name tenplex-compile \
    -t kungfu.azurecr.io/mw-megatron-lm-23.06:latest \
    python compile_kernels.py

docker commit tenplex-compile kungfu.azurecr.io/mw-megatron-lm-23.06-kernels:latest
docker push kungfu.azurecr.io/mw-megatron-lm-23.06-kernels:latest
