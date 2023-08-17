#!/usr/bin/env -S sh -c 'docker build --rm -t kungfu.azurecr.io/mw-megatron-lm:latest -f $0 .'
FROM kungfu.azurecr.io/mw-pytorch1:latest

WORKDIR /workspace

RUN apt-get install ninja-build
RUN pip install \
    six \
    regex \
    pybind11 \
    packaging

RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    git checkout 23.05 && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    rm -r /workspace/apex

# Megatron-LM
COPY . Megatron-LM
WORKDIR /workspace/Megatron-LM
