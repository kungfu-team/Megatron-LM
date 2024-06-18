#!/bin/sh
set -e

./Dockerfile-update

# docker run -i -t kungfu.azurecr.io/mw-megatron-lm-23.06-update:latest

./shell ./docker-main.sh
