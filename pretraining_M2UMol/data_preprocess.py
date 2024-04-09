
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import molepre
import pickle
from utils import *
import pandas as pd
import csv
import random
from tqdm import tqdm
import copy

import numpy as np
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)



class Data_class(Dataset):

    def __init__(self, triple):
        self.entity = triple#[:, 0]
    def __len__(self):
        return len(self.entity)
    def __getitem__(self, index):
        return  self.entity[index]


def load_data(args):


    def loadtrainvaltest():

        druglist=[]
        with open('data/drug_list.csv', encoding='utf-8') as f:
            for row in csv.reader(f):
                if row[0]!='drugbank_id':
                    druglist.append(row[0])


        train_pos_des=[]
        for i in range(len(druglist)):
                    train_pos_des.append(i)

        train_pos_des=np.array(train_pos_des)
        np.random.seed(1)
        np.random.shuffle(train_pos_des)
        train_pos = np.array(train_pos_des)

        train_data = train_pos
        return train_data,train_data,train_data

    train_data,_,_=loadtrainvaltest()
    params = {'batch_size': args.batch, 'shuffle': False, 'num_workers': args.workers, 'drop_last': False}  #True  True
    training_set = Data_class(train_data)
    train_loader = DataLoader(training_set, **params)


    print('Loading finished!')

    return train_loader


