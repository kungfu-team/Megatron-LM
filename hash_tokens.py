#!/usr/bin/env torchrun
import hashlib

import numpy as np
import torch
from tenplex.dataset import GPTDataset as TenplexGPTDataset

from megatron import arguments, get_args, get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.data.dataset_utils import (compile_helper,
                                         get_train_valid_test_split_)
from megatron.data.gpt_dataset import GPTDataset, get_indexed_dataset_
from megatron.initialize import initialize_megatron
from megatron.utils import get_ltor_masks_and_position_ids


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, tokenizer.eod, args.reset_position_ids,
        args.reset_attention_mask, args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def pytorch_dataset():
    data_prefix = "/data/dataset/gpt-2/my-gpt2_text_document"
    data_impl = "mmap"
    splits_string = "949,50,1"
    train_valid_test_num_samples = [1280000, 0, 0]
    seq_length = 1024
    seed = 1234
    skip_warmup = True
    return_doc_ids = False
    data_cache_path = None
    index = 0
    name = "train"
    do_shuffle = False

    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    print(splits)
    documents = np.arange(start=splits[index],
                          stop=splits[index + 1],
                          step=1,
                          dtype=np.int32)
    dataset = GPTDataset(name,
                         data_prefix,
                         documents,
                         indexed_dataset,
                         splits_string,
                         train_valid_test_num_samples[index],
                         seq_length,
                         seed,
                         return_doc_ids,
                         data_cache_path=data_cache_path,
                         do_shuffle=do_shuffle)
    return dataset


def hash_dataset(dataset, file_name: str):
    print(dataset.__class__)  # megatron.data.gpt_dataset.GPTDataset
    with open(file_name, "w", encoding="utf-8") as fi:
        for i, sample in enumerate(dataset):
            if i > 1024:
                break
            ha = hashlib.sha256()
            ha.update(sample["text"].tobytes())
            ha_str = str(ha.hexdigest())
            fi.write(ha_str + "\n")


def hash_batches(dataset, file_name: str):
    data_iterator = iter(dataset)
    # get_batch
    with open(file_name, "w") as fi:
        for i in range(1024):
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
                data_iterator)
            ha = hashlib.sha256()
            ha.update(tokens.tobytes())
            ha_str = str(ha.hexdigest())
            fi.write(ha_str + "\n")


def tenplex_dataset():
    mlfs_path = "/data/mlfs"
    jobid = "tenplex-samples"
    dp_rank = 0

    dataset = TenplexGPTDataset(mlfs_path, jobid, dp_rank)
    return dataset


def hash_pytorch():
    dataset = pytorch_dataset()
    hash_dataset(dataset, "/data/out/samples_pytorch.txt")
    # hash_batches(dataset, "/data/out/tokens_pytorch.txt")


def hash_tenplex():
    dataset = tenplex_dataset()
    hash_dataset(dataset, "/data/out/samples_tenplex.txt")
    # hash_batches(dataset, "/data/out/tokens_tenplex.txt")


def init():
    # arguments._print_args = noop
    # initialize_megatron(
    #     extra_args_provider=None,
    #     args_defaults={
    #         'tokenizer_type': 'GPT2BPETokenizer',
    #     },
    # )

    # torch.distributed.init_process_group()
    mpu.initialize_model_parallel(1, 1)
    compile_helper()


def main():
    init()

    hash_pytorch()
    hash_tenplex()


if __name__ == "__main__":
    main()
