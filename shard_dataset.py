import glob
import os

import numpy as np
import torch

from megatron import mpu
from megatron.data.dataset_utils import (build_train_valid_test_datasets,
                                         compile_helper)
from megatron.global_vars import _build_tokenizer


def save_shard(samples: dict, dp_rank: int, out_dir):
    if len(samples['text']) == 0:
        return

    text = np.concatenate(samples['text'])
    types = np.concatenate(samples['types'])
    labels = np.concatenate(samples['labels'])
    is_random = np.concatenate(samples['is_random'])
    loss_mask = np.concatenate(samples['loss_mask'])
    padding_mask = np.concatenate(samples['padding_mask'])
    truncated = np.concatenate(samples['truncated'])

    shard_num = samples['shard_num']
    print(f'DP rank {dp_rank}: save shard {shard_num}')
    print(f'DP rank {dp_rank}: text shape {text.shape}')

    rank_dir = os.path.join(out_dir, str(dp_rank))
    os.makedirs(rank_dir, exist_ok=True)
    path = os.path.join(rank_dir, f'samples_{shard_num:09d}.npz')
    np.savez(path,
             text=text,
             types=types,
             labels=labels,
             is_random=is_random,
             loss_mask=loss_mask,
             padding_mask=padding_mask,
             truncated=truncated)

    samples['shard_num'] += 1

    samples['text'] = []
    samples['types'] = []
    samples['labels'] = []
    samples['is_random'] = []
    samples['loss_mask'] = []
    samples['padding_mask'] = []
    samples['truncated'] = []


def shard_dataset(ds_dir: str, batch_size: int, dp_size: int, out_dir: str,
                  num_samples: int):
    device_batch_size = batch_size // dp_size

    samples = {}
    for dp_rank in range(dp_size):
        samples[dp_rank] = {}
        samples[dp_rank]['text'] = []
        samples[dp_rank]['types'] = []
        samples[dp_rank]['labels'] = []
        samples[dp_rank]['is_random'] = []
        samples[dp_rank]['loss_mask'] = []
        samples[dp_rank]['padding_mask'] = []
        samples[dp_rank]['truncated'] = []
        samples[dp_rank]['shard_num'] = 0

    sample_num = 0
    sample_files = glob.glob(os.path.join(ds_dir, 'samples_*'))
    for sample_file in sample_files:
        sample_shard = np.load(sample_file)
        for i in range(sample_shard['text'].shape[0]):
            batch_idx = sample_num % batch_size
            dp_rank = batch_idx // device_batch_size
            samples[dp_rank]['text'].append(sample_shard['text'][i])
            samples[dp_rank]['types'].append(sample_shard['types'][i])
            samples[dp_rank]['labels'].append(sample_shard['labels'][i])
            samples[dp_rank]['is_random'].append(sample_shard['is_random'][i])
            samples[dp_rank]['loss_mask'].append(sample_shard['loss_mask'][i])
            samples[dp_rank]['padding_mask'].append(
                sample_shard['padding_mask'][i])
            samples[dp_rank]['truncated'].append(sample_shard['truncated'][i])

            if len(samples[dp_rank]['text']) == num_samples:
                save_shard(samples[dp_rank], dp_rank, out_dir)

    for dp_rank in range(dp_size):
        save_shard(samples[dp_rank], dp_rank, out_dir)


def main():
    batch_size = 32
    dp_size = 2
    seq_len = 512
    num_samples = 16384
    full_ds_dir = f'/data/megatron-lm/bert/large_{seq_len}'
    out_dir = f'/data/megatron-lm/bert/shard{dp_size}_{seq_len}'
    shard_dataset(full_ds_dir, batch_size, dp_size, out_dir, num_samples)


if __name__ == '__main__':
    main()
