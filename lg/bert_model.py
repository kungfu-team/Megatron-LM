from megatron import get_args, initialize
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


def main():
    initialize._compile_dependencies = noop
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            'tokenizer_type': 'BertWordPieceLowerCase',
        },
    )

    m = build_model()
    print(m)


main()
