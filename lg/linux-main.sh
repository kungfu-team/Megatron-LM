#!/bin/sh
set -e

./shell

cd $(dirname $0)

. ./_x

# x ./deps.sh
# x ./bert.sh

echo "done $0"
