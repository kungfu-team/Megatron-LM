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
"""Processing data for pretraining."""

import argparse
import json
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import time

import numpy as np
import torch

try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.data import indexed_dataset
from megatron.tokenizer import build_tokenizer


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):

    def tokenize(self, *text):
        return text


class Encoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,
                    lang_vars=CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(json_line)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input',
                       type=str,
                       required=True,
                       help='Path to input JSON')
    group.add_argument(
        '--json-keys',
        nargs='+',
        default=['text'],
        help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences',
                       action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines',
                       action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type',
                       type=str,
                       required=True,
                       choices=[
                           'BertWordPieceLowerCase', 'BertWordPieceCase',
                           'GPT2BPETokenizer'
                       ],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file',
                       type=str,
                       default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file',
                       type=str,
                       default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod',
                       action='store_true',
                       help='Append an <eod> token to the end of a document.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix',
                       type=str,
                       required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl',
                       type=str,
                       default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers',
                       type=int,
                       required=True,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk-size',
                       type=int,
                       required=True,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--log-interval',
                       type=int,
                       default=100,
                       help='Interval between progress updates')
    group.add_argument('--sentences-dir', type=str)
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print(
                "Bert tokenizer detected, are you sure you don't want to split sentences?"
            )

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def main():
    args = get_args()

    batch_size = 32
    dp_degree = 2
    local_batch_size = batch_size // dp_degree

    tokenizer = build_tokenizer(args)
    vocab_size = tokenizer.vocab_size

    level = "document"
    if args.split_sentences:
        level = "sentence"

    sentence_paths = []
    for entry in os.scandir(args.sentences_dir):
        if entry.is_dir():
            for sentence_entry in os.scandir(entry.path):
                if sentence_entry.is_file() and sentence_entry.path.endswith(
                        '.npy'):
                    sentence_paths.append(sentence_entry.path)
    sentence_paths.sort()

    num_sentences = len(sentence_paths)

    dp_rank_sentence_paths = {}

    for dp_rank in range(dp_degree):
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in args.json_keys:
            output_bin_files[
                key] = f"{args.output_prefix}_{key}_{level}_{dp_rank:02d}.bin"
            output_idx_files[
                key] = f"{args.output_prefix}_{key}_{level}_{dp_rank:02d}.idx"
            builders[key] = indexed_dataset.make_builder(
                output_bin_files[key],
                impl=args.dataset_impl,
                vocab_size=vocab_size)

        # filter for each dp_rank
        offset = 0
        dp_rank_sentence_paths[dp_rank] = []
        for _ in range(num_sentences // batch_size):
            for i in range(local_batch_size):
                sample = offset + local_batch_size * dp_rank + i
                dp_rank_sentence_paths[dp_rank].append(sentence_paths[sample])
            offset = offset + batch_size

        proc_start = time.time()

        for key in args.json_keys:
            for i, sentence_path in enumerate(dp_rank_sentence_paths[dp_rank]):
                sentence = np.load(sentence_path)
                builders[key].add_item(sentence)
                builders[key].end_document()

                if i % args.log_interval == 0:
                    current = time.time()
                    elapsed = current - proc_start
                    print(f"Processed {i} sentences in {elapsed} s")

        for key in args.json_keys:
            builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    main()
