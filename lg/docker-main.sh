#!/bin/sh
set -e

cd $(dirname $0)

. ./_x

x ./bert.sh

echo "done $0"
