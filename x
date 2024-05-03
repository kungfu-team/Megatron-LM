#!/bin/sh
set -e

if [ -f /.dockerenv ]; then
    ./lg/docker-main.sh
else
    ./lg/fix.sh
    ./shell ./x
    # ./Dockerfile
    # ./lg/linux-main.sh
    # ./lg/bert.sh
    # ./lg/gpt.sh
fi

git add -A

echo "done $0"
