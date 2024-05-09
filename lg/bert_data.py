from megatron import arguments, get_args, initialize
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.model import BertModel


def noop(*_args, **_kvargs):
    pass


def main():
    initialize._compile_dependencies = noop
    arguments._print_args = noop
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            'tokenizer_type': 'BertWordPieceLowerCase',
        },
    )


main()
