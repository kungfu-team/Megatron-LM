#!/bin/sh

cd $(dirname $0)

_py_fix() {
    echo "fixing $1"
    isort $1
    yapf -i $1
}

py_fix() {
    for f in $@; do
        _py_fix $f
    done
}

py_fix ./hash_samples.py
py_fix ./lg/*.py
