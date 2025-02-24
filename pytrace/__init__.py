import os

pid = os.getpid()


class Counters(object):

    def __init__(self):
        self.counters = dict()

    def next(self, name):
        if name not in self.counters:
            self.counters[name] = 0
            return 0
        else:
            self.counters[name] += 1
            return self.counters[name]


_counters = Counters()


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


def dprint(msg):
    tab = indent * _ctx.depth
    print(tab + str(msg))


def f():
    print(1)


def noop(*_args, **_kvargs):
    pass


tf = open('trace.txt', 'w')


def putln(line):
    print(line)
    tf.write(line + '\n')


class TraceScope(object):

    def __init__(self, name=''):
        self.name = name
        self.ctx = _ctx

    def __enter__(self):
        depth = self.ctx.bgn()
        tab = indent * depth
        putln(tab + '{ // ' + self.name)

    def __exit__(self, a, b, c):
        depth = self.ctx.end()
        tab = indent * depth
        n = _counters.next(self.name)
        msg = tab + '} // ' + self.name + ' | ' + str(n)
        # msg += ' @ ' + str(pid)
        putln(msg)


def ptrace(name):
    print('[T] %s' % (name))


def with_trace(f):

    def g(*args, **kvargs):
        with TraceScope(f.__name__):
            return f(*args, **kvargs)

    g.__name__ = f.__name__
    return g


def with_trace_limit(f, limit):

    class G(object):

        def __init__(self):
            self.__name__ = f.__name__
            self.n = 0

        def __call__(self, *args, **kvargs):
            self.n += 1
            if self.n > limit:
                return f(*args, **kvargs)

            with TraceScope(f.__name__):
                return f(*args, **kvargs)

    g = G()
    return g


def traced_3(f):
    return with_trace_limit(f, 3)


def traced(f):
    return with_trace(f)


def log_args(fn, *args, **kvargs):
    for i, a in enumerate(args):
        putln('%s | [%d]: %s' % (fn, i, a))
    for k, v in kvargs:
        putln('%s | %s: %s' % (fn, k, v))


def with_log_args_limit(f, limit):

    class G(object):

        def __init__(self):
            self.n = 0
            self.__name__ = f.__name__

        def __call__(self, *args, **kvargs):
            self.n += 1
            if self.n < limit:
                log_args(f.__name__, *args, **kvargs)
            return f(*args, **kvargs)

    g = G()
    return g


def with_log_args(f):

    return with_log_args_limit(f, 3)
    # def g(*args, **kvargs):
    #     log_args(f.__name__, *args, **kvargs)
    #     return f(*args, **kvargs)

    # g.__name__ = f.__name__
    # return g


def with_log_unary(f):

    def g(x):
        putln('%s(%s)' % (f.__name__, x))
        return f(x)

    g.__name__ = f.__name__
    return g
