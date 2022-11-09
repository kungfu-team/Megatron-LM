#!/bin/bash
set -e

./Dockerfile-update

docker push kungfu.azurecr.io/mw-megatron-lm-update:latest
