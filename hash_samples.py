#!/usr/bin/env torchrun
import hashlib

import numpy as np
import torch
from tenplex.dataset import GPTDataset as TenplexGPTDataset

from megatron.core import mpu
from megatron.data.dataset_utils import (compile_helper,
                                         get_train_valid_test_split_)
from megatron.data.gpt_dataset import GPTDataset, get_indexed_dataset_


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

    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    documents = np.arange(start=splits[index], stop=splits[index + 1],
                          step=1, dtype=np.int32)
    dataset = GPTDataset(name, data_prefix, documents, indexed_dataset,
                         splits_string,
                         train_valid_test_num_samples[index],
                         seq_length, seed,
                         return_doc_ids,
                         data_cache_path=data_cache_path,
                         do_shuffle=do_shuffle)
    return dataset


def hash_dataset(dataset, file_name: str):
    with open(file_name, "w", encoding="utf-8") as fi:
        for i, sample in enumerate(dataset):
            if i > 1024:
                break
            ha = hashlib.sha256()
            ha.update(sample["text"].tobytes())
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


def hash_tenplex():
    dataset = tenplex_dataset()
    hash_dataset(dataset, "/data/out/samples_tenplex.txt")


def init():
    torch.distributed.init_process_group()
    mpu.initialize_model_parallel(1, 1)
    compile_helper()


def main():
    init()

    hash_pytorch()
    hash_tenplex()


if __name__ == "__main__":
    main()
