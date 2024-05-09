#!/bin/sh
set -e

cd $(dirname $0)

. ./_x

x ./bert.sh
cp -v trace.txt /log

echo "done $0"
