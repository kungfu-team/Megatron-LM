#!/usr/bin/env -S sh -c 'docker build --rm -t kungfu.azurecr.io/mw-megatron-lm:latest -f $0 .'

FROM kungfu.azurecr.io/mw-base:latest

# PORT
EXPOSE 6000

WORKDIR /workspace

RUN apt-get install ninja-build
RUN pip install \
    six \
    regex \
    pybind11

RUN git clone https://github.com/NVIDIA/apex.git
RUN cd apex && \
    git checkout 22.03 && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Megatron-LM
ADD . Megatron-LM
WORKDIR /workspace/Megatron-LM
