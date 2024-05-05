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


class Context(object):

    def __init__(self, name=''):
        self.name = name
        pass

    def __enter__(self):
        pass
        BGN(self.name)

    def __exit__(self, a, b, c):
        pass
        END(self.name)
