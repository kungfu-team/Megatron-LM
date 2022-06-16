#!/usr/bin/env -S sh -c 'docker build --rm -t kungfu.azurecr.io/mw-megatron-lm:latest -f $0 .'

FROM nvcr.io/nvidia/pytorch:22.05-py3

# PORT
EXPOSE 6000
EXPOSE 22

# NCCL
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_P2P_LEVEL=NVL

# Megatron-LM
RUN mkdir -p workspace
WORKDIR /workspace
ADD . Megatron-LM
WORKDIR /workspace/Megatron-LM
