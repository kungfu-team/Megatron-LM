#!/bin/sh
set -e

sudo rm -fr out
./Dockerfile-update

./mount-mlfs.sh

# ./shell ./run_hash.sh
./shell ./run_train_gpt.sh

# sha1sum out/*
