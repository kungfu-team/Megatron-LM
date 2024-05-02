#!/bin/sh
set -e

cd $(dirname $0)

. ./_x

x ./requirements.txt

x ./bert.sh

echo "done $0"
