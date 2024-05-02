#!/bin/sh
set -e

cd $(dirname $0)

ie https://download.pytorch.org/whl/torch_stable.html >torch.txt

echo "done $0"
