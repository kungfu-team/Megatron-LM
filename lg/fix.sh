#!/bin/sh

cd $(dirname $0)

py_fix() {
    isort $1
    yapf -i $1
}

py_fix ./*.py
py_fix ../pytrace/*.py
py_fix ../shadow/*.py
py_fix ../a.py
py_fix ../samples_mapping.py

cd ..
# mv ./log/dos2unix ./log/dos2unix.py
py_fix ./log/dos2unix
# mv ./log/dos2unix.py ./log/dos2unix

if [ $(uname) = "Darwin" ]; then
    cxx-fix megatron/data/build_mapping_impl.cpp
fi
