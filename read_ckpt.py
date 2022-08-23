import torch

rank = 0
step = 400
pp_rank = 0
mp_rank = 0
path = f"/data/marcel/training/{rank}/ckpt/iter_{step:07d}/mp_rank_{mp_rank:02d}_{pp_rank:03d}/model_optim_rng.pt"
ckpt = torch.load(path)

args = ckpt["args"]
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
