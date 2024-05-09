import torch
from megatron import arguments, get_args, initialize
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.model import BertModel


def noop(*_args, **_kvargs):
    pass


def build_model(pre_process=True, post_process=True):

    args = get_args()
    args.model_type = ModelType.encoder_or_decoder
    num_tokentypes = 2 if args.bert_binary_head else 0
    model = BertModel(
        num_tokentypes=num_tokentypes,
        add_binary_head=args.bert_binary_head,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process,
    )

    return model


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


def pp(p: torch.nn.parameter.Parameter):
    return show_tensor(p)


def main():
    initialize._compile_dependencies = noop
    arguments._print_args = noop
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            'tokenizer_type': 'BertWordPieceLowerCase',
        },
    )

    m = build_model()
    # print(m)
    print(m.__class__)
    # for x in dir(m):
    #     print(x)
    for p in m.parameters():
        print('%s' % (pp(p)))


main()
