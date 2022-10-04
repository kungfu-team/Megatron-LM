import torch

rank = 0
step = 20
pp_rank = 0
tp_rank = 1
#  path = f"/data/marcel/training/{rank}/ckpt/iter_{step:07d}/mp_rank_{mp_rank:02d}_{pp_rank:03d}/model_optim_rng.pt"
#  path = f"/data/marcel/training/{rank}/ckpt/global_step{step}/mp_rank_{tp_rank:02d}_model_states.pt"
#  path = f"/data/marcel/training/{rank}/ckpt/global_step{step}/layer_00-model_00-model_states.pt"
#  path = "/data/marcel/transformed/mp_rank_00_model_states.pt"
path = "/data/marcel/united/mp_rank_00_model_states.pt"
ckpt = torch.load(path)

#  args = ckpt["args"]
#  for arg in vars(args):
#      print(f"{arg}: {getattr(args, arg)}")

#  print(ckpt.keys())
#  print(ckpt)

import pprint

pprint.pprint(ckpt)
