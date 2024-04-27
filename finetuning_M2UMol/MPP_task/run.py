import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
import csv
import copy
import pandas as pd
import numpy as np


tasklist=['bace', 'bbbp', 'clintox', 'hiv', 'muv', 'tox21', 'sider', 'toxcast']
ft_hyper_list = ['64_0.0005_0.1_0.1_0.0005_0.0002', '32_0.0001_0.1_0.2_0.0002_0.0002', '64_0.0005_0.2_0.1_0.0005_0.0002',
                 '256_0.0001_0.1_0.1_0.0002_0.0001', '128_0.0001_0.1_0.1_0.005_0.0005', '64_0.0001_0.2_0.1_0.001_0.0005',
                 '256_0.0005_0.2_0.2_0.0005_0.0002', '256_0.0001_0.1_0.1_0.0005_0.0002']
result_file = 'result.txt'
pretrained_file = './pretrained_model/pre-trained_M2UMol.pt'


for i in range(len(tasklist)):
    hyper_list = ft_hyper_list[i].split("_")
    bs = hyper_list[0]
    wd = hyper_list[1]
    dp = hyper_list[2]
    attdp = hyper_list[3]
    lr = hyper_list[4]
    jtlr = hyper_list[5]

    for k in range(3):
        os.system('python train.py --dataset ' + str(tasklist[i]) + ' --runseed ' + str(k)+
                  ' --batch_size ' + str(bs)+' --mweight_decay ' + str(wd) +
                  ' --mdropout ' + str(dp)+' --mattdropout ' + str(attdp) +
                  ' --mlr ' + str(lr) + ' --jingtiaolr ' + str(jtlr) + ' --result_file ' + result_file +
                  ' --pretrained_file ' + str(pretrained_file))
