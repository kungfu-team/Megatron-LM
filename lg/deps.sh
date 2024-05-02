#!/bin/sh
set -e

cd $(dirname $0)

. ./_x

# NVIDIA-SMI 550.54.14
# Driver Version: 550.54.14
# CUDA Version: 12.4

x ./requirements.torch.txt
x ./requirements.txt
x ./check.py

echo "done $0"
