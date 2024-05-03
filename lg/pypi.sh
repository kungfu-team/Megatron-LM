#!/bin/sh
set -e

cd $(dirname $0)

# url=https://download.pytorch.org/whl/cu118
url=https://download.pytorch.org/whl/torch_stable.html

ie $url >torch.txt

echo "done $0"
