#!/bin/sh
set -e

cd $(dirname $0)

. ./_x

x ./jsonl2mmid.sh
# x ./bert.sh

echo "done $0"
