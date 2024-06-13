import argparse
import hashlib
import io

import numpy as np
import torch

from megatron.core import mpu
from megatron.data.dataset_utils import (compile_helper,
                                         get_train_valid_test_split_)
from megatron.data.gpt_dataset import GPTDataset, get_indexed_dataset_


def create_dataset(seq_length: int):
    data_prefix = "/data/dataset/gpt-2/my-gpt2_text_document"
    data_impl = "mmap"
    splits_string = "949,50,1"
    train_valid_test_num_samples = [1280000, 0, 0] # args.train_iters * args.global_batch_size
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



def save_npzs(ds, out_dir: str):
    npzs_file = open(f'{out_dir}/samples.npzs', 'wb')
    indices_file = open(f'{out_dir}/indices.txt', 'wt')

    indices_file.write('1\n')
    indices_file.write('/path/to/bin NUMBEROFSAMPES\n')

    current_offset = 0
    for _, sample in enumerate(ds):
        buff = io.BytesIO()
        np.savez(buff, text=sample['text'])

        val = buff.getvalue()
        npzs_file.write(val)
        new_offset = current_offset + len(val)
        indices_file.write(f'{current_offset} {new_offset}\n')
        current_offset = new_offset

        buff.close()

    npzs_file.close()
    indices_file.close()


def main():
    seq_length = 1024
    # save_dir = "/data/datset/gpt-2/enwiki/npzs_seq{seq_length}"
    save_dir = "/data/datset/gpt-2/enwiki/npzs_seq{seq_length}_new"

    # init distribution
    torch.distributed.init_process_group()
    mpu.initialize_model_parallel(1, 1)

    # compile dataset helper
    compile_helper()

    ds = create_dataset(seq_length)
    save_npzs(ds, save_dir)


if __name__ == '__main__':
    main()
