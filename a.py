#!/usr/bin/env python3

from pytrace import with_trace as tr

def traced(f):
    return tr(f)
    # return f


@traced
def f():
    print(1)


f = tr(f)

f()
