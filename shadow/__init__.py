import torch


def show_dtype(t: torch.dtype):
    names = {
        torch.float16: 'f16',
        torch.float32: 'f32',
    }
    return names.get(t, '?')


def show_shape(s: torch.Size):
    dims = []
    for d in s:
        dims.append(int(d))
    return '[%s]' % (','.join(str(d) for d in dims))


def show_tensor(x: torch.Tensor):
    return '%s%s' % (show_dtype(x.dtype), show_shape(x.shape))


def show_parameter(p: torch.nn.parameter.Parameter):
    return show_tensor(p)
