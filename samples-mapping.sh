#!/bin/sh
set -e

python3 ./samples_mapping.py $(cat samples-mapping.txt)
