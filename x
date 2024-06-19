#!/bin/sh
set -e

sudo rm -fr out
sudo systemctl restart mlfs
./hash_samples.sh
./Dockerfile-update
./shell ./run_hash.sh
sha1sum out/*
