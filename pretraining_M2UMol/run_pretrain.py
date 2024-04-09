import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from data_preprocess import load_data
from train import train_model
import os
import argparse
import random
import molepre
from layer import M2UMol
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
    parser.add_argument('--out_file', default='result.txt',
                        help='Path to data result file. e.g., result.txt')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate. Default is 1e-3.')
    parser.add_argument('--encoder_dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability). Default is 0.5.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability). Default is 0.1.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='num_layers of 2D encoder')
    parser.add_argument('--weight_decay', default=0,
                        help='Weight decay (L2 loss on parameters) Default is 0.')#5e-4
    parser.add_argument('--batch', type=int, default=16, #
                        help='Batch size. Default is 32.')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs to train. Default is 150.')#
    parser.add_argument('--tem', type=float, default=1.0,
                        help='The temperature coefficient in contrastive learning loss. Default is 1')
    parser.add_argument('--mcls_loss_ratio', type=float, default=0.5,
                        help='Ratio of modalitiy classification task. Default is 1')
    parser.add_argument('--dimensions', type=int, default=128,
                        help='dimensions of feature. Default is 128.')
    parser.add_argument('--hidden1', default=128,
                        help='Number of hidden units for encoding layer. Default is 128.')
    parser.add_argument('--output_name', default='M2UMol',
                        help='The name of the saved pre-trained M2UMol')
    args = parser.parse_args()

    return args


def Create_model(args):
    model = M2UMol(num_layers=args.num_layers,hidden1=args.hidden1,encoder_dropout=args.encoder_dropout,dropout=args.dropout)
    ignored_params1 = list(map(id, model.encoderText.parameters()))
    ignored_params2 = list(map(id, model.encoder3D.parameters()))
    ignored_params3 = list(map(id, model.encoder2D.parameters()))
    ignored_params = ignored_params1+ignored_params2+ignored_params3
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer = Adam([
        {'params': model.encoder3D.parameters(), 'lr': 5e-4, 'weight_decay': 1e-9},
        {'params': model.encoderText.parameters(), 'lr': 2e-5, 'weight_decay': 1e-9},
        {'params': model.encoder2D.parameters(), 'lr': 1e-3, 'weight_decay': 1e-9},
        {'params': base_params, 'lr': 5e-4, 'weight_decay': 1e-9},
    ], lr=args.lr, weight_decay=args.weight_decay)
    scheduler_1 = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.8)
#
    return model, optimizer,scheduler_1
args = settings()

args.cuda = not args.no_cuda and torch.cuda.is_available()


train_loader = load_data(args)
model, optimizer,sch = Create_model(args)
train_model(model, optimizer, sch,train_loader,args)

