def BGN(name):
    print(name)


def END(name):
    print(name)


def ptrace(name):
    print('[T] %s' % (name))


def traced(f):

    def g(*args, **kvargs):
        BGN('BGN')
        f(*args, **kvargs)
        END('END')

    return g


def f():
    print(1)
