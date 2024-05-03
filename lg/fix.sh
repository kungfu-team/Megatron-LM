#!/bin/sh

cd $(dirname $0)

py_fix() {
    isort $1
    yapf -i $1
}

py_fix ./pretrain_bert.py
py_fix ../pytrace/*.py
