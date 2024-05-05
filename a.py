#!/usr/bin/env python3

from pytrace import traced


@traced
def f():
    print(1)


@traced
def main():
    f()


main()
