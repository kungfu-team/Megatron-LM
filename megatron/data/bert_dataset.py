# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT Style dataset."""

import glob
import os
import re

import numpy as np
import torch
from megatron import get_args, get_tokenizer, mpu
from megatron.data.dataset_utils import (create_masked_lm_predictions,
                                         create_tokens_and_tokentypes,
                                         get_a_and_b_segments,
                                         get_samples_mapping,
                                         truncate_segments)


class BertDataset(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, data_prefix, num_epochs,
                 max_num_samples, masked_lm_prob, max_seq_length,
                 short_seq_prob, seed, binary_head):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.binary_head = binary_head

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(
            self.indexed_dataset,
            data_prefix,
            num_epochs,
            max_num_samples,
            self.max_seq_length - 3,  # account for added tokens
            short_seq_prob,
            self.seed,
            self.name,
            self.binary_head)

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))
        return build_training_sample(
            sample,
            seq_length,
            self.max_seq_length,  # needed for padding
            self.vocab_id_list,
            self.vocab_id_to_token_dict,
            self.cls_id,
            self.sep_id,
            self.mask_id,
            self.pad_id,
            self.masked_lm_prob,
            np_rng,
            self.binary_head)


class BertDatasetBatchFile(torch.utils.data.Dataset):

    def __init__(self,
                 name,
                 indexed_dataset,
                 data_prefix,
                 num_epochs,
                 max_num_samples,
                 masked_lm_prob,
                 max_seq_length,
                 short_seq_prob,
                 seed,
                 binary_head,
                 samples_per_file=16384,
                 batch_size=32):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.binary_head = binary_head

        self.samples_per_file = samples_per_file
        dir_path = os.path.dirname(data_prefix)
        self.batch_files = glob.glob(dir_path + '/batch*')
        self.batch_files.sort()

        self.dp_degree = mpu.get_data_parallel_world_size()
        self.do_shard = self.dp_degree > 1
        if self.do_shard:
            self.dp_rank = mpu.get_data_parallel_rank()
            self.device_batch_size = batch_size // self.dp_degree

        self.loaded_files = {}

        last_file = self.batch_files[-1]
        file_name = os.path.basename(last_file)
        mat = re.match(r'batch_(\d+).npz', file_name)
        if mat is not None:
            batch_num = int(mat.group(1))
        else:
            raise ValueError('match is None')
        self.num_samples = samples_per_file * batch_num
        last_samples = np.load(last_file)
        last_batch_num_samples = last_samples['text'].shape[0]
        self.num_samples = self.num_samples + last_batch_num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.do_shard:
            batch = idx // self.device_batch_size
            device_batch = (self.dp_degree + self.dp_rank) * batch
            device_batch = device_batch + idx % self.device_batch_size
            idx = device_batch

        sample_idx = idx % self.samples_per_file
        file_idx = idx // self.samples_per_file

        if self.batch_files[file_idx] in self.loaded_files:
            samples = self.loaded_files[self.batch_files[file_idx]]
        else:
            samples = np.load(self.batch_files[file_idx])
            self.loaded_files[self.batch_files[file_idx]] = samples

        train_sample = {
            'text': samples['text'][sample_idx],
            'types': samples['types'][sample_idx],
            'labels': samples['labels'][sample_idx],
            'is_random': int(samples['is_random'][sample_idx]),
            'loss_mask': samples['loss_mask'][sample_idx],
            'padding_mask': samples['padding_mask'][sample_idx],
            'truncated': int(samples['truncated'][sample_idx])
        }

        return train_sample


def build_training_sample(sample, target_seq_length, max_seq_length,
                          vocab_id_list, vocab_id_to_token_dict, cls_id,
                          sep_id, mask_id, pad_id, masked_lm_prob, np_rng,
                          binary_head):
    """Biuld training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
    """

    if binary_head:
        # We assume that we have at least two sentences in the sample
        assert len(sample) > 1
    assert target_seq_length <= max_seq_length

    # Divide sample into two segments (A and B).
    if binary_head:
        tokens_a, tokens_b, is_next_random = get_a_and_b_segments(
            sample, np_rng)
    else:
        tokens_a = []
        for j in range(len(sample)):
            tokens_a.extend(sample[j])
        tokens_b = []
        is_next_random = False

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = truncate_segments(tokens_a, tokens_b, len(tokens_a),
                                  len(tokens_b), max_num_tokens, np_rng)

    # Build tokens and toketypes.
    tokens, tokentypes = create_tokens_and_tokentypes(tokens_a, tokens_b,
                                                      cls_id, sep_id)

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _,
     _) = create_masked_lm_predictions(tokens, vocab_id_list,
                                       vocab_id_to_token_dict, masked_lm_prob,
                                       cls_id, sep_id, mask_id,
                                       max_predictions_per_seq, np_rng)

    # Padding.
    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np \
        = pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length)

    train_sample = {
        'text': tokens_np,
        'types': tokentypes_np,
        'labels': labels_np,
        'is_random': int(is_next_random),
        'loss_mask': loss_mask_np,
        'padding_mask': padding_mask_np,
        'truncated': int(truncated)
    }
    return train_sample


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                               dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np
