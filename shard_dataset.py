import glob
import os
from typing import Dict

import numpy as np
import torch

from megatron import mpu
from megatron.data.dataset_utils import (build_train_valid_test_datasets,
                                         compile_helper)
from megatron.global_vars import _build_tokenizer


def save_shard(samples: dict, dp_rank: int, out_dir: str,
               samples_per_shard: int, keys: list):
    total_num_samples = samples['text'].shape[0]
    if total_num_samples == 0:
        return

    num_shards = total_num_samples // samples_per_shard
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = start_idx + samples_per_shard

        new_shard: Dict[str, np.ndarray] = {}
        for key in keys:
            new_shard[key] = samples[key][start_idx:end_idx]

        rank_dir = os.path.join(out_dir, str(dp_rank))
        os.makedirs(rank_dir, exist_ok=True)
        path = os.path.join(rank_dir, f'samples_{shard_idx:09d}.npz')
        np.savez(path, **new_shard)

        print(f'finished shard {shard_idx}')


def concat(samples, key, tail):
    if key in samples:
        return np.concatenate((samples[key], tail))
    return tail


def shard_dataset(ds_dir: str, batch_size: int, dp_size: int, out_dir: str,
                  samples_per_shard: int, keys: list):
    device_batch_size = batch_size // dp_size

    samples: Dict[int, dict] = {}
    for dp_rank in range(dp_size):
        samples[dp_rank] = {}

    sample_files = glob.glob(os.path.join(ds_dir, 'samples_*'))

    for i, sample_file in enumerate(sample_files):
        sample_shard = np.load(sample_file)
        shard_num_samples = sample_shard['text'].shape[0]

        for dp_rank in range(dp_size):
            # create indices
            dp_indices = []
            num_batches = shard_num_samples // batch_size
            for batch in range(num_batches):
                start_idx = batch * batch_size
                start_idx = start_idx + dp_rank * device_batch_size
                end_idx = start_idx + device_batch_size
                indices = list(range(start_idx, end_idx))
                dp_indices.extend(indices)

            for key in keys:
                samples[dp_rank][key] = concat(samples[dp_rank], key,
                                               sample_shard[key][dp_indices])

        print(f'finished file {i}')

    for dp_rank in range(dp_size):
        save_shard(samples[dp_rank], dp_rank, out_dir, samples_per_shard, keys)


def main():
    batch_size = 32
    dp_size = 2
    seq_len = 512
    samples_per_shard = 16384
    full_ds_dir = f'/data/megatron-lm/bert/large_{seq_len}'
    out_dir = f'/data/megatron-lm/bert/shard{dp_size}_seq{seq_len}'
    keys = [
        'text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask',
        'truncated'
    ]
    shard_dataset(full_ds_dir, batch_size, dp_size, out_dir, samples_per_shard,
                  keys)


if __name__ == '__main__':
    main()
