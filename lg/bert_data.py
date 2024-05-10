import megatron
import numpy as np
import torch
from megatron import arguments, get_args, initialize
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.initialize import initialize_megatron
from megatron.training import build_train_valid_test_data_loaders
from pytrace import dprint as print
from pytrace import traced, with_log_args
from shadow import show_tensor


def noop(*_args, **_kvargs):
    pass


@traced
@with_log_args
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


def show_npa(a: np.ndarray):
    return 'npa<%s%s>' % (a.dtype, a.shape)


def show_tuple(x):
    for k, v in sorted(x.items()):
        if isinstance(v, np.ndarray):
            print('- {}: {}'.format(k, show_npa(v)))
        elif isinstance(v, int):
            print('- {}: {}'.format(k, v))
        else:
            print('- {}: {}'.format(k, v.__class__))


def show_item(x):
    for k, v in sorted(x.items()):
        print('- {}: {}'.format(k, show_tensor(v)))
    print('.')


@traced
def show_ds(it: torch.utils.data.dataloader.DataLoader):
    assert (isinstance(it, torch.utils.data.dataloader.DataLoader))
    # print(it.__class__)
    print(len(it))
    for x in it:
        show_item(x)
        break
    print(len(it))


# AttributeError: module 'megatron.data' has no attribute 'bert_dataset'. Did you mean: 'vit_dataset'?
from megatron.data.bert_dataset import BertDataset

BDS = megatron.data.bert_dataset.BertDataset


def show_bert_ds(ds: BertDataset):
    # megatron.data.bert_dataset.BertDataset
    #  megatron.data.bert_dataset.BertDataset):
    print(ds.__class__)  # megatron.data.bert_dataset.BertDataset
    for x in ds:
        # show_item(x)
        # print(x)
        # print(x.__class__)
        show_tuple(x)
        break


@traced
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
    print(args.dataloader_type)  # single

    a, b, c = train_valid_test_datasets_provider([800, 80, 80])
    show_bert_ds(a)
    show_bert_ds(b)
    show_bert_ds(c)

    # train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_loaders(
    #     train_valid_test_datasets_provider)

    # show_ds(train_data_iterator)
    # show_ds(valid_data_iterator)
    # show_ds(test_data_iterator)
    print(megatron.data.bert_dataset.BertDataset)
    # print(megatron.data)


main()
