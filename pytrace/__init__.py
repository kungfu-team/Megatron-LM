def BGN(name):
    print(name)


def END(name):
    print(name)


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


def ptrace(name):
    print('[T] %s' % (name))


def with_trace(f):

    def g(*args, **kvargs):
        with Context(f.__name__):
            return f(*args, **kvargs)

    return g


def traced(f):
    return with_trace(f)
