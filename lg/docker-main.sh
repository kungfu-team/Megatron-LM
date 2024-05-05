#!/bin/sh
set -e

cd $(dirname $0)

. ./_x

x ./bert.sh
cat trace.txt

echo "done $0"
