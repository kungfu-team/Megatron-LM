#!/bin/sh
set -e
cd $(dirname $0)

. ./_x

x ./requirements.txt
x ./requirements.torch.txt
x ./check.py

echo "done $0"
