def BGN(name):
    print(name)


def END(name):
    print(name)


def ptrace(name):
    print('[T] %s' % (name))


def traced(f, name=''):

    def g(*args, **kvargs):
        BGN('BGN ' + name)
        f(*args, **kvargs)
        END('END ' + name)

    return g


def f():
    print(1)


def noop(*args, **kvargs):
    pass
