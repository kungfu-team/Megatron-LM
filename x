#!/bin/sh
set -e

cd $(dirname $0)

. ./lg/measure.sh

main() {

    if [ -f /.dockerenv ]; then
        measure ./lg/docker-main.sh
    else
        measure ./lg/fix.sh
        measure ./a.py
        measure ./samples-mapping.sh
        git add -A
        exit
        if [ $(uname) = "Linux" ]; then
            measure ./shell ./x
        fi
        # ./Dockerfile
        # ./lg/linux-main.sh
        # ./lg/bert.sh
        # ./lg/gpt.sh
    fi
}

measure main

git add -A

echo "done $0"
