import argparse
import hashlib
import io
import os

import numpy as np
import torch

from megatron import mpu
from megatron.data.dataset_utils import (build_train_valid_test_datasets,
                                         compile_helper)
from megatron.global_vars import _build_tokenizer


def create_train_ds(prefix: str, train_val_test_num_samples, seq_length: int):
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=[prefix],
        data_impl='infer',
        splits_string='949,50,1',
        train_valid_test_num_samples=train_val_test_num_samples,
        max_seq_length=seq_length,
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


#  train_sample = {
#      'text': tokens_np,
#      'types': tokentypes_np,
#      'labels': labels_np,
#      'is_random': int(is_next_random),
#      'loss_mask': loss_mask_np,
#      'padding_mask': padding_mask_np,
#      'truncated': int(truncated)}
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


def save_npzs(ds, out_dir: str):
    npzs_file = open(f'{out_dir}/samples.npzs', 'wb')
    indices_file = open(f'{out_dir}/indices.txt', 'wt')

    indices_file.write('1\n')
    indices_file.write('/path/to/bin NUMBEROFSAMPES\n')

    current_offset = 0
    for i, sample in enumerate(ds):
        buff = io.BytesIO()
        np.savez(buff,
                 text=sample['text'],
                 types=sample['types'],
                 labels=sample['labels'],
                 is_random=sample['is_random'],
                 loss_mask=sample['loss_mask'],
                 padding_mask=sample['padding_mask'],
                 truncated=sample['truncated'])

        val = buff.getvalue()
        npzs_file.write(val)
        new_offset = current_offset + len(val)
        indices_file.write(f'{current_offset} {new_offset}\n')
        current_offset = new_offset

        buff.close()

    npzs_file.close()
    indices_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-length', type=int)
    parser.add_argument('--tokenizer-type',
                        type=str,
                        default='BertWordPieceLowerCase')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument(
        '--vocab-file',
        type=str,
        default='/data/megatron-lm/bert/bert-large-uncased-vocab.txt')
    parser.add_argument('--vocab-extra-ids', type=int, default=0)
    parser.add_argument('--make-vocab-size-divisible-by',
                        type=int,
                        default=128)
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1)
    args = parser.parse_args()

    batch_size = 32
    eval_iters = 10
    train_iters = 4000
    seq_length = args.seq_length

    train_samples = train_iters * batch_size
    train_val_test_num_samples = [
        train_samples, eval_iters * batch_size, eval_iters * batch_size
    ]

    # init distribute
    torch.distributed.init_process_group('nccl')
    mpu.initialize_model_parallel(1, 1, 1, 0)

    compile_helper()

    _build_tokenizer(args)

    #  prefix = '/data/megatron-lm/bert/bert_text_sentence'
    prefix = '/data/openwebtext/bert/bert_text_sentence'
    full_train_ds = create_train_ds(prefix, train_val_test_num_samples,
                                    seq_length)
    #  save_dir = f'/data/megatron-lm/bert/npzs_seq{seq_length}'
    save_dir = f'/data/megatron-lm/bert/openwebtext/npzs_seq{seq_length}'
    save_npzs(full_train_ds, save_dir)


if __name__ == '__main__':
    main()
