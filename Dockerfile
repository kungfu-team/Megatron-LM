#!/usr/bin/env -S sh -c 'docker build --rm -t kungfu.azurecr.io/mw-megatron-lm:latest -f $0 .'

FROM nvcr.io/nvidia/pytorch:22.05-py3

# PORT
EXPOSE 6000

# Megatron-LM
RUN mkdir -p workspace
WORKDIR /workspace
ADD . Megatron-LM
WORKDIR /workspace/Megatron-LM
