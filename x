#!/bin/sh
set -e

./lg/bert.sh
./lg/gpt.sh

git add -A

echo "done $0"
