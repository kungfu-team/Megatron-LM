#!/bin/sh
set -e

cd $(dirname $0)

. ./_x

x ./check.py
# exit

x ./requirements.txt
x ./requirements.torch.txt
exit

x ./bert.sh

echo "done $0"
