import networkx as nx
from torch_geometric.data import Data,Batch
from torch.utils.data import Dataset, DataLoader
import molepre
import pickle
from utils import *
import pandas as pd
import csv
import random
from tqdm import tqdm
import copy
#from  molepre import create_graph_data
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


class Data_class_ddi(Dataset):

    def __init__(self, triple,MOL_EDGE_LIST_FEAT_MTX):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.relationtype=triple[:,2]
        self.MOL_EDGE_LIST_FEAT_MTX=MOL_EDGE_LIST_FEAT_MTX
        #self.label = triple[:, 3]

    def __len__(self):
        return len(self.relationtype)

    def __getitem__(self, index):

        return  (self.entity1[index], self.entity2[index], self.relationtype[index])

    def collate_fn(self, batch):

        def create_graph_data(id):
            edge_index = self.MOL_EDGE_LIST_FEAT_MTX[id][0]
            features = self.MOL_EDGE_LIST_FEAT_MTX[id][1]
            edge_attr1 = self.MOL_EDGE_LIST_FEAT_MTX[id][2]

            return Data(x=features, edge_index=edge_index, edge_attr=edge_attr1, ids=id)

        hdata=[]
        tdata=[]
        relation=[]
        for ind, (h, t,r) in enumerate(batch):
                hdata.append(create_graph_data(h))
                tdata.append(create_graph_data(t))
                relation.append(r)

        pos_h = Batch.from_data_list(hdata)
        pos_t = Batch.from_data_list(tdata)


        return pos_h,pos_t,relation

def load_data_ddi_unseen(args):
    import pickle

    with open("alldrugdata.pkl", 'rb') as f:  # read
        MOL_EDGE_LIST_FEAT_MTX = pickle.load(f)
    #print(MOL_EDGE_LIST_FEAT_MTX)
    drug_list = []
    with open('data/drug_list.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] != 'drugbank_id':
                drug_list.append(row[0])


    seed=args.seed

    def loadtrainvaltest1(args):
        #train dataset
        if args.split=='cold':
            train=pd.read_csv('data/'+str(seed)+'/ddi_trainingdaseen3fold.csv') # ddi_trainingdaseen3fold-str.csv' for scaffold split
        if args.split=='scaffold':
            train=pd.read_csv('data/'+str(seed)+'/ddi_trainingdaseen3fold-str.csv')
        train_pos=[(h, t, r) for h, t, r in zip(train['d1'], train['d2'], train['type'])]
        np.random.shuffle(train_pos)
        train_pos = np.array(train_pos)
        train_pos1=np.zeros((train_pos.shape[0],train_pos.shape[1]), dtype=int)
        for i in range(train_pos.shape[0]):
            train_pos1[i][0] = int(drug_list.index(train_pos[i][0]))
            train_pos1[i][1] = int(drug_list.index(train_pos[i][1]))
            train_pos1[i][2] = int(train_pos[i][2])
        label_list=[]
        for i in range(train_pos.shape[0]):
            label=np.zeros((86))
            label[int(train_pos[i][2])]=1
            label_list.append(label)
        label_list=np.array(label_list).astype('int')
        train_data= np.concatenate([train_pos1, label_list],axis=1)

        #val dataset
        if args.split == 'cold':
            val = pd.read_csv('data/'+str(seed)+'/ddi_testdaunun3fold.csv')  #ddi_testdaunun3fold-str.csv
        if args.split == 'scaffold':
            val = pd.read_csv('data/' + str(seed) + '/ddi_testdaunun3fold-str.csv')  # ddi_testdaunun3fold-str.csv
        val_pos = [(h, t, r) for h, t, r in zip(val['d1'], val['d2'], val['type'])]
        #np.random.seed(args.seed)
        np.random.shuffle(val_pos)
        val_pos= np.array(val_pos)
        val_pos1 = np.zeros((val_pos.shape[0], val_pos.shape[1]), dtype=int)
        for i in range(val_pos.shape[0]):
            val_pos1[i][0] = int(drug_list.index(val_pos[i][0]))
            val_pos1[i][1] = int(drug_list.index(val_pos[i][1]))
            val_pos1[i][2] = int(val_pos[i][2])
        label_list = []
        for i in range(val_pos.shape[0]):
            label = np.zeros((86))
            label[int(val_pos[i][2])] = 1
            label_list.append(label)
        label_list = np.array(label_list).astype('int')
        val_data = np.concatenate([val_pos1, label_list], axis=1)

        #test dataset
        if args.split == 'cold':
            test = pd.read_csv('data/'+str(seed)+'/ddi_testdaunun3fold.csv')   #ddi_testdaunun3fold-str.csv
        if args.split == 'scaffold':
            test = pd.read_csv('data/' + str(seed) + '/ddi_testdaunun3fold-str.csv')
        test_pos = [(h, t, r) for h, t, r in zip(test['d1'],test['d2'], test['type'])]
        np.random.shuffle(test_pos)
        test_pos= np.array(test_pos)
        test_pos1 = np.zeros((test_pos.shape[0], test_pos.shape[1]), dtype=int)

        for i in range(test_pos.shape[0]):
            test_pos1[i][0] = int(drug_list.index(test_pos[i][0]))
            test_pos1[i][1] = int(drug_list.index(test_pos[i][1]))
            test_pos1[i][2] = int(test_pos[i][2])
        label_list = []
        for i in range(test_pos.shape[0]):
            label = np.zeros((86))
            label[int(test_pos[i][2])] = 1
            label_list.append(label)
        label_list = np.array(label_list).astype('int')
        test_data = np.concatenate([test_pos1, label_list], axis=1)
        #print(train_data)
        print('loading train val test...')
        print(train_data.shape)
        print(val_data.shape)
        print(test_data.shape)
        return train_data,val_data,test_data

    train_data,val_data,test_data=loadtrainvaltest1(args)#
    params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': args.workers, 'drop_last': False}

    class DrugDataLoader(DataLoader):
        def __init__(self, data, **kwargs):
            super().__init__(data, collate_fn=data.collate_fn, **kwargs)

    training_set = Data_class_ddi(train_data,MOL_EDGE_LIST_FEAT_MTX)

    train_loader = DrugDataLoader(training_set, **params)


    validation_set = Data_class_ddi(val_data,MOL_EDGE_LIST_FEAT_MTX)

    val_loader = DrugDataLoader(validation_set, **params)


    test_set = Data_class_ddi(test_data,MOL_EDGE_LIST_FEAT_MTX)

    test_loader = DrugDataLoader(test_set, **params)


    print('Loading finished!')
    return train_loader, val_loader, test_loader
