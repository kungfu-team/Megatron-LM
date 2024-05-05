import sys

import numpy as np


def show(x):
    return '%s%s' % (x.dtype, x.shape)


def f(filename):
    print(filename)
    x = np.load(filename)
    print(show(x))
    pass


def main(filenames):
    for filename in filenames:
        f(filename)


main(sys.argv[1:])
