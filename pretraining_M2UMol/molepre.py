import itertools
from collections import defaultdict
from operator import neg
import random
import math
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
import pickle
import csv
from torch_scatter import scatter
from chem import BOND_TYPES, mol_to_smiles
import copy
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)

################yuanyou
def edge_features(bond):
    # bond_type = bond.GetBondType()
    # return torch.tensor([
    #     bond_type == Chem.rdchem.BondType.SINGLE,
    #     bond_type == Chem.rdchem.BondType.DOUBLE,
    #     bond_type == Chem.rdchem.BondType.TRIPLE,
    #     bond_type == Chem.rdchem.BondType.AROMATIC,
    #     bond.GetIsConjugated(),
    #     bond.IsInRing()]).long()
    # bond_type = bond.GetBondType()
    return torch.tensor(bond_to_feature_vector(bond), dtype=torch.long)#.long()

# def edge_features(bond):
#     # bond_type = bond.GetBondType()
#     # return torch.tensor([
#     #     bond_type == Chem.rdchem.BondType.SINGLE,
#     #     bond_type == Chem.rdchem.BondType.DOUBLE,
#     #     bond_type == Chem.rdchem.BondType.TRIPLE,
#     #     bond_type == Chem.rdchem.BondType.AROMATIC,
#     #     bond.GetIsConjugated(),
#     #     bond.IsInRing()]).long()
#     bond_type = bond.GetBondType()
#     return torch.tensor(bond_to_feature_vector(bond), dtype=torch.long)#.long()



def get_mol_edge_list_and_feat_mtx(mol_graph, smiles,atom_symbols):
    features = [(atom.GetIdx(), torch.tensor(atom_to_feature_vector(atom), dtype=torch.long)) for atom in mol_graph.GetAtoms()]
    features.sort()  # to make sure that the feature matrix is aligned according to the idx of the atom
    _, features = zip(*features)
    features = torch.stack(features)
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),*edge_features(b)) for b in mol_graph.GetBonds()])
    #edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].long()) if len(edge_list) else (
    torch.LongTensor([]), torch.LongTensor([]))
    #torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats
    undirected_edge_list = edge_list

    return undirected_edge_list.T, features, edge_feats

################yuanyou
# def get_mol_edge_list_and_feat_mtx(mol_graph, smiles,atom_symbols):
#     features = [(atom.GetIdx(), atom_features(atom,atom_symbols)) for atom in mol_graph.GetAtoms()]
#     features.sort()  # to make sure that the feature matrix is aligned according to the idx of the atom
#     _, features = zip(*features)
#     features = torch.stack(features)
#     # print(222222222222222222)
#     # print(features.shape)
#     edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),*edge_features(b)) for b in mol_graph.GetBonds()])
#     edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
#     torch.LongTensor([]), torch.FloatTensor([]))
#     edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
#     edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats
#     undirected_edge_list = edge_list

#     return undirected_edge_list.T, features, edge_feats


def rdmol_to_data(mol, smiles=None, data_cls=Data):


    N = mol.GetNumAtoms()
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    row, col = edge_index
    d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1)  # (num_edge, 1)

    data = data_cls(z=z, pos=pos, edge_index=edge_index, edge_type=edge_type,edge_length=d,
                rdmol=copy.deepcopy(mol), smiles=smiles)

    return data

wu2d=[]
wu3d=[]
mask = np.load('mask.npy')

######## load molecules
print('loading molecules...')
druglist=[]
with open('data/drug_list.csv',encoding='utf-8') as f:
    for row in csv.reader(f):
        if row[0]!='drugbank_id':
            druglist.append(row[0])



######## load molecular descriptions
print('loading molecular descriptions...')
drugsum = []
drugsup = []
with open('data/description-sup.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        drugsum.append(row[0])
        drugsup.append(row[1])


##########load molecular descriptions
print('loading molecular structures...')
sdfs_2d = Chem.SDMolSupplier("data/2Dstructures.sdf",removeHs=False)
sdfs_3d = Chem.SDMolSupplier("data/3Dstructures.sdf",removeHs=False)

conflist=[]
druglist_3d=[]
for mol in sdfs_3d:
    if mol is not None:
        con=rdmol_to_data(mol)
        conflist.append(con)
        id = mol.GetProp('DRUGBANK_ID')
        druglist_3d.append(id)
# maxatom=0    ##compute max atom num
# for i in range(len(conflist)):
#     molzi = conflist[i].rdmol
#     typelist=[]
#     if torch.max(conflist[i].z) >=maxatom:
#         maxatom=torch.max(conflist[i].z)


print('loading molecular 2D topology graphs and 3D conformers...')
druglist_2d=[]
conformer2d_ori=[]
for mol in sdfs_2d:
    if mol is not None:
        id = mol.GetProp('DRUGBANK_ID')
        druglist_2d.append(id)
        try:
            smiles = mol.GetProp('SMILES')
            conformer2d_ori.append(smiles)
        except:
            wu2d.append(id) #Record the molecules without smiles
            conformer2d_ori.append('CCC(=O)N(CC(C)N(C)CCC1=CC=CC=C1)C1=CC=CC=C1')  #Keep the model working properly, using arbitrary smiles replacements, and will not be used in pretraining

drugsmils=[]
drugsmils1=[]
drugsmils3=[]
for i in range(len(druglist)):
    drugsmils.append(druglist[i])
    try:
        drugsmils1.append(conformer2d_ori[druglist_2d.index(druglist[i])])
    except:
        wu2d.append(druglist[i])
        drugsmils1.append('CCC(=O)N(CC(C)N(C)CCC1=CC=CC=C1)C1=CC=CC=C1')

    if druglist[i] in druglist_3d:
        drugsmils3.append(conflist[druglist_3d.index(druglist[i])])
    else:
        wu3d.append(druglist[i])
        drugsmils3.append(conflist[0])

for i in range(len(drugsmils1)):
    try:
        Chem.MolFromSmiles(drugsmils1[i].strip()).GetNumAtoms()
    except:
        if druglist[i]=='DB00515':
            drugsmils1[i] = 'N.N.Cl[Pt+2]Cl'  #Some molecules that cannot be converted to mol format can be set manually
        else:
            drugsmils1[i]='CCC(=O)N(CC(C)N(C)CCC1=CC=CC=C1)C1=CC=CC=C1' #In order to ensure that the model runs, arbitrary data substitution is used, which will not be used in training due to the mask mechanism

drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip()), smiles.strip()) for id, smiles in zip(drugsmils, drugsmils1)]


# ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup]) #max atom
# print(999999999999)
# print(ATOM_MAX_NUM)

# symbols = list()   #save symbol,Using the same symbol during the pre-training and fine-tuning phases allows information such as the atomic characteristics of the molecule to be consistent
# drug_smile_dict = {}
# for mol in drug_id_mol_graph_tup:
#             symbols.extend(atom.GetSymbol() for atom in mol[1].GetAtoms())
# symbols = list(set(symbols))
# np.save('symbo.npy',symbols)

a=np.load('symbo.npy')  #load symbols
symbols=a.tolist()
MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol,smiles,symbols)
                                for drug_id, mol,smiles in drug_id_mol_graph_tup}
MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}
TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])


def create_graph_data(id):
    edge_index = MOL_EDGE_LIST_FEAT_MTX[id][0]
    features = MOL_EDGE_LIST_FEAT_MTX[id][1]
    edge_attr1=MOL_EDGE_LIST_FEAT_MTX[id][2]

    return Data(x=features, edge_index=edge_index, edge_attr=edge_attr1, ids=id)

def return2d(idx):
    pos_h_samples1 = []
    ids23d = idx
    druglistbatch=[]
    for i in range(ids23d.shape[0]):
        druglistbatch.append(druglist[ids23d[i]])
    for h in druglistbatch:
        h_data1 = create_graph_data(h)
        pos_h_samples1.append(h_data1)
    pos_h_samples1 = Batch.from_data_list(pos_h_samples1)
    pos_tri2d = pos_h_samples1
    return pos_tri2d
def return2d3d(idx):
    pos_h_samples = []
    pos_h_samples1 = []
    ids23d = idx
    keep=[]
    for kkk in range(ids23d.shape[0]):
        if mask[ids23d[kkk]][2] == 1 and mask[ids23d[kkk]][3] == 1:  # and self.mask[kkk][2]==1:
            keep.append(ids23d[kkk])
    keep=np.array(keep)
    ids23d=keep
    druglistbatch=[]
    for i in range(ids23d.shape[0]):
        druglistbatch.append(druglist[ids23d[i]])
    for h in druglistbatch:
        h_data1 = create_graph_data(h)
        h_data = drugsmils3[druglist.index(h)]
        pos_h_samples.append(h_data)
        pos_h_samples1.append(h_data1)
    pos_h_samples = Batch.from_data_list(pos_h_samples)
    pos_h_samples1 = Batch.from_data_list(pos_h_samples1)
    pos_tri3d = pos_h_samples
    pos_tri2d = pos_h_samples1
    return pos_tri2d,pos_tri3d


def returntext(idx):
    idstext = idx
    keep = []
    for kkk in range(idstext.shape[0]):
        if mask[idstext[kkk]][0] == 1 and mask[idstext[kkk]][2] == 1:  # and self.mask[kkk][2]==1:
            keep.append(idstext[kkk])
    keep = np.array(keep)
    idstext = keep
    drugsumbatch=[]
    for i in range(idstext.shape[0]):
        drugsumbatch.append(drugsum[idstext[i]])
    druglistbatchtext = []
    for i in range(idstext.shape[0]):
        druglistbatchtext.append(druglist[idstext[i]])
    pos_h_samples1text = []
    for h in druglistbatchtext:
        h_data1 = create_graph_data(h)
        pos_h_samples1text.append(h_data1)
    pos_h_samples1 = Batch.from_data_list(pos_h_samples1text)
    pos_tri2d = pos_h_samples1

    return pos_tri2d,drugsumbatch

def returnbio(idx):
    targetfea = np.load('data/targetfea.npy')
    enzymefea = np.load('data/enzymefea.npy')
    drugcatefea = np.load('data/drugcatefea.npy')
    targetfea = torch.from_numpy(targetfea)
    enzymefea = torch.from_numpy(enzymefea)
    drugcatefea = torch.from_numpy(drugcatefea)
    ids1 = idx
    keep = []
    for kkk in range(ids1.shape[0]):
        if mask[ids1[kkk]][2] == 1  and mask[ids1[kkk]][4]+mask[ids1[kkk]][5]+mask[ids1[kkk]][6]!=0:
            keep.append(ids1[kkk])
    keep = np.array(keep)
    ids1 = keep
    druglistbatchbio = []
    for i in range(ids1.shape[0]):
        druglistbatchbio.append(druglist[ids1[i]])
    pos_h_samples1bio = []
    for h in druglistbatchbio:
        h_data1 = create_graph_data(h)
        pos_h_samples1bio.append(h_data1)
    pos_h_samples1 = Batch.from_data_list(pos_h_samples1bio)
    pos_tri2d = pos_h_samples1

    return [pos_tri2d,targetfea[ids1],enzymefea[ids1],drugcatefea[ids1]]
