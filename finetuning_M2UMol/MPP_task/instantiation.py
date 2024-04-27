from torch.optim import Adam
from layer import MMDDI2d_256#,MMDDI2d_1024,SA_DDI  #CSGNN,MMDDI
import numpy as np
import os
import random
import torch

def Create_model_ddi_2d(args,num_tasks):

    model = MMDDI2d_256(dropout=args.mdropout,args=args,num_tasks=num_tasks)
    model1 = torch.compile(model)

    if args.traintype=='freeze':
        ignored_params1 = list(map(id, model1.mlp1.parameters()))
        ignored_params2 = list(map(id, model1.attnfea.parameters()))
        ignored_params = ignored_params1 + ignored_params2 #+ ignored_params3
        base_params1 = filter(lambda p: id(p) not in ignored_params, model1.parameters())

        optimizer = Adam([
            {'params': model1.mlp1.parameters(), 'lr': args.mlr, 'weight_decay': args.mweight_decay},
            {'params': model1.attnfea.parameters(), 'lr': args.mlr, 'weight_decay': args.mweight_decay},
            {'params': base_params1, 'lr': args.jingtiaolr, 'weight_decay': args.mweight_decay},
        ], lr=args.mlr, weight_decay=args.mweight_decay)
    else:
        optimizer = Adam(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
    return model, optimizer