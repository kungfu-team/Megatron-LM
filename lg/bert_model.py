import torch
from megatron import get_args, mpu
from megatron.core.enums import ModelType
from megatron.initialize import initialize_megatron
from megatron.model import BertModel
from megatron.training import setup_model_and_optimizer


def build_model(pre_process=True, post_process=True):

    args = get_args()
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
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            'tokenizer_type': 'BertWordPieceLowerCase',
        },
    )
    # model_type = ModelType.encoder_or_decoder
    # model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
    #     model_provider, model_type)

    # torch.distributed.init_process_group('nccl')
    # mpu.initialize_model_parallel(1, 1, 1, 0)

    m = build_model()
    print(m)


main()
