import collections
import json
import os

import torch


def create_value_dict(value):
    if value is None:
        return None
    if isinstance(value, (collections.OrderedDict, dict)):
        elements = {}
        for key, val in value.items():
            elements[key] = create_value_dict(val)
        return elements
    if isinstance(value, torch.Tensor):
        return {'tensor': list(value.size())}
    if isinstance(value, (list, set, tuple)):
        elements = {}
        for i, val in enumerate(value):
            elements[i] = create_value_dict(val)
        return elements
    if isinstance(value, (int, float, str, bool)):
        return value

    return 'NotPrimitiveNorTensor'


def create_ckpt_dict(ckpt):
    elements = {}

    for key, val in ckpt.items():
        elements[key] = create_value_dict(val)

    return elements


def megatron_lm():
    base_dir = "/data/marcel/training"
    size = 4
    pp = 4
    mp = 1
    dp = size // (pp * mp)
    step = 10
    out_dir = os.path.join(
        os.path.expanduser('~'),
        "Elasticity/Repo/transformer-checkpoint/megatron-lm")
    out_dir = os.path.join(out_dir, f"pp{pp:02d}/mp{mp:02d}/dp{dp:02d}")
    os.makedirs(out_dir, exist_ok=True)

    for rank in range(size):
        rank_input_path = os.path.join(base_dir,
                                       f"{rank}/ckpt/iter_{step:07d}")

        if not os.path.isdir(rank_input_path):
            continue

        rank_output_path = os.path.join(out_dir, f"rank{rank:02d}")
        os.makedirs(rank_output_path)

        for entry in os.scandir(rank_input_path):
            if not entry.is_dir():
                continue

            ckpt_path = os.path.join(entry.path, "model_optim_rng.pt")

            ckpt = torch.load(ckpt_path)
            ckpt_dict = create_ckpt_dict(ckpt)
            new_name = entry.name.split(".")[0] + ".json"
            out_path = os.path.join(rank_output_path, new_name)

            with open(out_path, "w") as dict_file:
                json.dump(ckpt_dict, dict_file, indent=4)


def main():
    megatron_lm()


if __name__ == "__main__":
    main()
