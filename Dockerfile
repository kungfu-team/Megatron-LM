#!/usr/bin/env -S sh -c 'docker build --rm -t `cat tag.txt` -f $0 .'

FROM kungfu.azurecr.io/mw-megatron-lm-23.06-update:latest

WORKDIR /src
ADD . .
