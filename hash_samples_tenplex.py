import hashlib

import numpy as np
import torch
from tenplex.dataset import GPTDataset


def hash_samples():
    mlfs_path = "/data/mlfs"
    jobid = "tenplex-samples"
    dp_rank = 0

    dataset = GPTDataset(mlfs_path, jobid, dp_rank)

    with open("/data/out/samples_tenplex.txt", "w", encoding="utf-8") as fi:
        for i, sample in enumerate(dataset):
            if i > 128:
                break
            ha = hashlib.sha256()
            ha.update(sample["text"].tobytes())
            ha_str = str(ha.hexdigest())
            fi.write(ha_str + "\n")

def main():
    hash_samples()

if __name__ == "__main__":
    main()
