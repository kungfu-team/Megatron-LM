import torch


def show_dtype(t: torch.dtype):
    names = {
        torch.float16: 'f16',
        torch.float32: 'f32',
    }
    return names.get(t, '?')


def shape_dims(s: torch.Size):
    dims = []
    for d in s:
        dims.append(int(d))
    return [d for d in dims]


def show_dims(dims):
    return ','.join(str(x) for x in dims)


def show_tensor(x: torch.Tensor):
    return '%s[%s]' % (show_dtype(x.dtype), show_dims(shape_dims(x.shape)))


def show_parameter(p: torch.nn.parameter.Parameter):
    return show_tensor(p)


def show_model(m: torch.nn.Module):
    for p in m.parameters():
        print('%s' % (show_parameter(p)))


def prod(ns):
    from functools import reduce
    from operator import mul
    return reduce(mul, ns, 1)


def flip(m):
    return sorted((v, k) for k, v in m.items())


def scalar_size(t):
    sizes = {
        'f16': 2,
        'f32': 4,
    }
    return sizes[t]


def stat_model(m: torch.nn.Module):
    by_type = dict()
    by_scalar = dict()
    tot = 0
    for x in m.parameters():
        t, dims = show_dtype(x.dtype), shape_dims(x.shape)
        k = '%s[%s]' % (t, show_dims(dims))
        by_type[k] = by_type.get(k, 0) + 1

        n = prod(dims)
        by_scalar[t] = by_scalar.get(t, 0) + n
        tot += scalar_size(t) * n

    for n, k in flip(by_type):
        print('{} x {:,}'.format(k, n))

    for n, k in flip(by_scalar):
        print('{} x {:,}'.format(k, n))

    print('tot: {:,}'.format(tot))
