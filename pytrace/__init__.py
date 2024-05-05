class Context(object):

    def __init__(self):
        self.depth = 0

    def bgn(self):
        depth = self.depth
        self.depth += 1
        return depth

    def end(self):
        self.depth -= 1
        return self.depth


_ctx = Context()

indent = ' ' * 4


def f():
    print(1)


def noop(*_args, **_kvargs):
    pass


class TraceScope(object):

    def __init__(self, name=''):
        self.name = name
        self.ctx = _ctx

    def __enter__(self):
        depth = self.ctx.bgn()
        tab = indent * depth
        print(tab + '{ //' + self.name)

    def __exit__(self, a, b, c):
        depth = self.ctx.end()
        tab = indent * depth
        print(tab + '} //' + self.name)


def ptrace(name):
    print('[T] %s' % (name))


def with_trace(f):

    def g(*args, **kvargs):
        with TraceScope(f.__name__):
            return f(*args, **kvargs)

    return g


def traced(f):
    return with_trace(f)
