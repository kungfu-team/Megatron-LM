#!/usr/bin/env -S sh -c 'docker build --rm -t `cat tag.txt` -f $0 .'

FROM ubuntu:22.04
