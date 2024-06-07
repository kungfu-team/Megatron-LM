#!/usr/bin/env -S sh -c 'docker build --rm -t `cat tag.txt` -f $0 .'

# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# FROM  kungfu.azurecr.io/mw-pytorch2:latest
FROM kungfu.azurecr.io/mw-megatron-lm-23.06-update:latest
# FROM nvcr.io/nvidia/pytorch:24.04-py3

# /usr/local/lib/python3.10/dist-packages/torch/utils/cpp_extension.py:1968: UserWarning: TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation.

RUN python3 -m pip install nltk

WORKDIR /src
ADD . .

ENV PYTHONPATH=/src
ENTRYPOINT []
