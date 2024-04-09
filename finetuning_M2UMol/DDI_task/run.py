import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
import csv
import copy
import pandas as pd

import numpy as np

def set_random_seed(seed=0, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=False)

bs=256
wd=0.0002
dp=0.7
lr=0.0005
for k in range(3):
    os.system('python main.py --seed ' + str(k) + ' --lr ' + str(lr) + ' --batch ' + str(bs) + ' --weight_decay ' + str(wd) + ' --dropout ' + str(
        dp))
