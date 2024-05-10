import torch
from megatron import arguments, get_args, initialize
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.initialize import initialize_megatron
from megatron.training import build_train_valid_test_data_iterators
from pytrace import dprint as print
from pytrace import traced
from shadow import show_tensor


def noop(*_args, **_kvargs):
    pass


def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        short_seq_prob=args.short_seq_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        binary_head=args.bert_binary_head,
    )

    return train_ds, valid_ds, test_ds


def show_item(x):
    for k, v in sorted(x.items()):
        print('- {}: {}'.format(k, show_tensor(v)))
    print('.')


@traced
def show_ds(it: torch.utils.data.DataLoader):
    print(len(it))
    # print(it.__mro__)
    assert (isinstance(
        it, torch.utils.data.dataloader._MultiProcessingDataLoaderIter))
    # assert (isinstance(it, torch.utils.data.DataLoader)) # failed
    for x in it:
        show_item(x)
        break
    print(len(it))


def main():
    initialize._compile_dependencies = noop
    arguments._print_args = noop
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            'tokenizer_type': 'BertWordPieceLowerCase',
        },
    )

    args = get_args()
    args.iteration = 0
    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider)

    show_ds(train_data_iterator)
    show_ds(valid_data_iterator)
    show_ds(test_data_iterator)


main()
