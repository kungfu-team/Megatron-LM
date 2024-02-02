#!/usr/bin/env -S sh -c 'docker build --rm -t kungfu.azurecr.io/mw-megatron-lm-23.06:latest -f $0 .'
FROM kungfu.azurecr.io/mw-pytorch2:latest

WORKDIR /workspace

RUN apt-get install ninja-build
RUN pip install --no-cache-dir \
    six \
    regex \
    pybind11 \
    packaging \
    tensorboard

RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    git checkout 23.06 && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /workspace

COPY . Megatron-LM
WORKDIR /workspace/Megatron-LM
RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir ./build/tenplex
