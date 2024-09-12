#!/bin/bash
set -e

docker pull kungfu.azurecr.io/mw-megatron-lm-23.06-tenplex:latest

./Dockerfile-update

docker push kungfu.azurecr.io/mw-megatron-lm-23.06-update:latest
