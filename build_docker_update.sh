#!/bin/bash
set -e

./Dockerfile-update
docker push kungfu.azurecr.io/mw-megatron-lm-23.06-debug:latest
