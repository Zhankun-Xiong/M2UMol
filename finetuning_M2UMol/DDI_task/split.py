# import pandas as pd
# import numpy as py
# import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
import csv
import copy
import pandas as pd

import numpy as np


import os
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, cut_tree
from matplotlib import pyplot as plt
from collections import Counter


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
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

drug_list = []
smiles=[]
with open('data/drug_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            drug_list.append(row[0])
            smiles.append(row[1])

# print(len(drug_list))
#random.seed(0)
#random.shuffle(drug_list)
for k in range(3):
     #os.system('python main.py --aggregator GCN --feature_type uniform --in_file data/DTI.edgelist --out_file test.txt --zhongzi '+str(k))

    vec_list = []
    for smi in smiles:
         m1 = Chem.MolFromSmiles(smi)
         fp4 = list(AllChem.GetMorganFingerprintAsBitVect(m1, radius=2, nBits=256))
         vec_list.append(fp4)

    Z = linkage(vec_list, 'average', metric='jaccard')
    cluster_num = round(len(vec_list) * 0.5)
    cluster = cut_tree(Z, cluster_num).ravel()
     #plt.hist(cluster)
    # print(cluster)
    # print(Counter(cluster).items())
    # print(sorted(Counter(cluster).items(), key=lambda item: item[1], reverse=True))
    stat_dict = {k: v for k, v in sorted(Counter(cluster).items(), key=lambda item: item[1], reverse=True)}
    traindrug=[]
    valdrug = []
    testdrug = []
    data_dict = defaultdict(list)
    for i in stat_dict.keys():

         pos = np.nonzero(cluster == i)[0]
         #print(pos)
         if len(traindrug) < round((len(smiles) *2)//3):
             traindrug += list(pos)
             continue
         # if len(traindrug) == round(len(smiles) * 0.7) and len(valdrug) < round(len(smiles) * 0.1):
         #     valdrug += list(pos)
         #     continue
         else:
             testdrug += list(pos)
             continue

    alllink=pd.read_csv('data/newddi.csv')
    #
    #testdrug=drug_list[k*len(drug_list)//3:(k+1)*len(drug_list)//3]
    #traindrug=drug_list[0:k*len(drug_list)//3]+drug_list[(k+1)*len(drug_list)//3:]
    print(len(traindrug))
    # print(len(valdrug))
    print(len(testdrug))


    for i in range(len(traindrug)):
        traindrug[i]=drug_list[int(traindrug[i])]
    for i in range(len(valdrug)):
        valdrug[i] = drug_list[int(valdrug[i])]
    for i in range(len(testdrug)):
        testdrug[i] = drug_list[int(testdrug[i])]
    trainlink=[]
    vallinkunun = []
    vallinkunseen = []
    testlinkunun = []
    testlinkunseen = []
    print(alllink.shape[0])
    for ii in range(alllink.shape[0]):
        print(ii)
        if alllink.loc[ii]['d1'] in traindrug and alllink.loc[ii]['d2'] in traindrug:
            trainlink.append(alllink.loc[ii])

        # if alllink.loc[ii]['d1'] in traindrug and alllink.loc[ii]['d2'] in valdrug:
        #     vallinkunseen.append(alllink.loc[ii])
        # if alllink.loc[ii]['d1'] in valdrug and alllink.loc[ii]['d2'] in traindrug:
        #     vallinkunseen.append(alllink.loc[ii])
        # if alllink.loc[ii]['d1'] in valdrug and alllink.loc[ii]['d2'] in valdrug:
        #     vallinkunun.append(alllink.loc[ii])
        # if alllink.loc[ii]['d1'] in traindrug and alllink.loc[ii]['d2'] in testdrug:
        #     testlinkunseen.append(alllink.loc[ii])
        # if alllink.loc[ii]['d1'] in testdrug and alllink.loc[ii]['d2'] in traindrug:
        #     testlinkunseen.append(alllink.loc[ii])
        if alllink.loc[ii]['d1'] in testdrug and alllink.loc[ii]['d2'] in testdrug:
            testlinkunun.append(alllink.loc[ii])
    #print(trainlink.shape)
    #print(testlink)
    #
    #
    #
    columns = ['d1', 'type', 'd2']
    #
    df1 = pd.DataFrame(columns = columns, data = trainlink)
    print(df1.shape)
    df1.to_csv('data/'+str(k)+'/ddi_trainingdaseen3fold-str.csv', index=None,encoding='utf-8')



    # df3 = pd.DataFrame(columns = columns, data = testlinkunseen)
    # print(df3.shape)
    # df3.to_csv('data/'+str(k)+'/ddi_testdaunseen1-str.csv', index=None, encoding='utf-8')

    df4 = pd.DataFrame(columns=columns, data=testlinkunun)
    print(df4.shape)
    df4.to_csv('data/' + str(k) + '/ddi_testdaunun3fold-str.csv', index=None, encoding='utf-8')

    # df3 = pd.DataFrame(columns=columns, data=vallinkunseen)
    # print(df3.shape)
    # df3.to_csv('data/' + str(k) + '/ddi_valdaunseen1.csv', index=None, encoding='utf-8')

    # df4 = pd.DataFrame(columns=columns, data=vallinkunun)
    # print(df4.shape)
    # df4.to_csv('data/' + str(k) + '/ddi_valdaunun1-str.csv', index=None, encoding='utf-8')







##DB00919	76	DB01337
