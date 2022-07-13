import hashlib
import os

import numpy as np
import torch

from megatron import mpu
from megatron.data.dataset_utils import (build_train_valid_test_datasets,
                                         compile_helper)
from megatron.global_vars import _build_tokenizer


def create_train_ds(prefix: str, train_val_test_num_samples):
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=[prefix],
        data_impl='infer',
        splits_string='949,50,1',
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=1024,
        masked_lm_prob=0.15,
        short_seq_prob=0.1,
        seed=1234,
        skip_warmup=True,
        binary_head=True)

    return train_ds


def hash_head(ds, num):
    for i, sample in enumerate(ds):
        b = sample['text'].tobytes()
        h = hashlib.sha256(b).hexdigest()
        print(f'{i:02d}: {h}')

        if i >= num:
            break


def load_samples(ds, num):
    for i, sample in enumerate(ds):
        if i >= num:
            break


def save_batches(ds, num_samples=16384):
    samples = []
    shard_num = 0
    for i, sample in enumerate(ds):
        if i > 0 and i % num_samples == 0:
            # save
            text = np.concatenate([[x['text']] for x in samples])
            types = np.concatenate([[x['types']] for x in samples])
            labels = np.concatenate([[x['labels']] for x in samples])
            is_random = np.concatenate([[x['is_random']] for x in samples])
            loss_mask = np.concatenate([[x['loss_mask']] for x in samples])
            padding_mask = np.concatenate([[x['padding_mask']]
                                           for x in samples])
            truncated = np.concatenate([[x['truncated']] for x in samples])

            print(f'save shard {shard_num}')

            path = '/data/megatron-lm/bert/large_1024'
            path = os.path.join(path, f'samples_{shard_num:09d}.npz')
            np.savez(path,
                     text=text,
                     types=types,
                     labels=labels,
                     is_random=is_random,
                     loss_mask=loss_mask,
                     padding_mask=padding_mask,
                     truncated=truncated)

            shard_num = shard_num + 1
            samples = []

        samples.append(sample)


def main():
    batch_size = 32
    dp_degree = 2
    device_batch_size = batch_size // dp_degree
    eval_iters = 10
    train_iters = 4000

    full_ds_dir = '/data/megatron-lm/bert/bert_text_sentence'
    full_train_ds = create_train_ds(prefix, train_val_test_num_samples)
    save_batches(full_train_ds)


if __name__ == '__main__':
    main()
