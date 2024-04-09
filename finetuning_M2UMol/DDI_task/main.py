import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
from data_preprocess import load_data_ddi_unseen
from train import train_model_ddi_2d
from torch.optim import Adam
from layer import M2UMol
import os
import random
import molepre
import argparse
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=False)

def settings():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
    parser.add_argument('--workers', type=int, default=0,help='Number of parallel workers. Default is 0.')
    parser.add_argument('--lr', type=float, default=1e-3,help='Initial learning rate. Default is 5e-4.')#2e-5  1e-3  3e-4
    parser.add_argument('--dropout', type=float, default=0.3,help='Dropout rate (1 - keep probability). Default is 0.5.')
    parser.add_argument('--attdropout', type=float, default=0.5,help='Dropout rate (1 - keep probability). Default is 0.5.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay (L2 loss on parameters) Default is 5e-4.')#7e-4 1e-9
    parser.add_argument('--batch', type=int, default=256,help='Batch size. Default is 256.')
    parser.add_argument('--epochs', type=int, default=100,help='Number of epochs to train. Default is 30.')#
    parser.add_argument('--seed', default=0,help='Number of seed.')
    parser.add_argument('--split', default='scaffold',help='split settings')
    args = parser.parse_args()

    return args
args = settings()

args.cuda = not args.no_cuda and torch.cuda.is_available()

def Create_model(args):

    model = M2UMol(args=args)
    model1=torch.compile(model)

    ignored_params1 = list(map(id, model1.mlp1.parameters()))
    ignored_params2 = list(map(id, model1.attnfea.parameters()))
    ignored_params = ignored_params1 + ignored_params2 #+ ignored_params3
    base_params1 = filter(lambda p: id(p) not in ignored_params, model1.parameters())
    optimizer = Adam([
        {'params': model1.mlp1.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': model1.attnfea.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': base_params1, 'lr': 1e-4, 'weight_decay': args.weight_decay},
    ], lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


train_loader, val_loader, test_loader = load_data_ddi_unseen(args)
model_ddi, optimizer_ddi = Create_model(args)

train_model_ddi_2d(model_ddi, optimizer_ddi, train_loader, val_loader, test_loader,args)
