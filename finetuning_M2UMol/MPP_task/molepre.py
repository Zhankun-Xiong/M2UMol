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
import pickle
import csv
from torch_scatter import scatter
from chem import BOND_TYPES, mol_to_smiles
import copy
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# f = open('data/samples_all_ddi.pkl', 'rb')
# aa = pickle.load(f)
# conflist=[]
# for data in aa:
#    conflist.append(data)
#print(conflist)
"""
allowable_features = {
    'possible_atomic_num_list':       list(range(1, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

"""

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

"""
def edge_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()

def get_mol_edge_list_and_feat_mtx(mol_graph, smiles,atom_symbols):
    # print(111111111111111111111111)
    # print(mol_graph.shape)
    # print(smiles)
    features = [(atom.GetIdx(), atom_features(atom,atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()  # to make sure that the feature matrix is aligned according to the idx of the atom
    _, features = zip(*features)
    features = torch.stack(features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),*edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
    torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats
    # edge_attr = (bond_attr(MolFromSmiles(smiles)))
    # edge_attr = torch.FloatTensor(edge_attr)
    undirected_edge_list = edge_list#torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list

    # assert TOTAL_ATOM_FEATS == features.shape[-1], "Expected atom n_features and retrived n_features not matching"
    return undirected_edge_list.T, features, edge_feats
"""


from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
def edge_features(bond):
    # bond_type = bond.GetBondType()
    # return torch.tensor([
    #     bond_type == Chem.rdchem.BondType.SINGLE,
    #     bond_type == Chem.rdchem.BondType.DOUBLE,
    #     bond_type == Chem.rdchem.BondType.TRIPLE,
    #     bond_type == Chem.rdchem.BondType.AROMATIC,
    #     bond.GetIsConjugated(),
    #     bond.IsInRing()]).long()
    bond_type = bond.GetBondType()
    return torch.tensor(bond_to_feature_vector(bond), dtype=torch.long)#.long()



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





"""

## graphmvp fenlei


def edge_features(bond):
    # bond_type = bond.GetBondType()
    # return torch.tensor([
    #     bond_type == Chem.rdchem.BondType.SINGLE,
    #     bond_type == Chem.rdchem.BondType.DOUBLE,
    #     bond_type == Chem.rdchem.BondType.TRIPLE,
    #     bond_type == Chem.rdchem.BondType.AROMATIC,
    #     bond.GetIsConjugated(),
    #     bond.IsInRing()]).long()
    bond_type = bond.GetBondType()
    return torch.tensor([allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())], dtype=torch.long)#.long()



def get_mol_edge_list_and_feat_mtx(mol_graph, smiles,atom_symbols):
    features = [(atom.GetIdx(), torch.tensor([allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())], dtype=torch.long)) for atom in mol_graph.GetAtoms()]
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


"""











wu2d=[]
wu3d=[]
mask = np.load('mask.npy')
drugsum = []
drugsup = []

drug_list = []
with open('data/drug_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            drug_list.append(row[0])
drug_list_smiles = []
with open('data/drug_list.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != 'drugbank_id':
            drug_list_smiles.append(row[1])
#drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip()), smiles.strip()) for id, smiles in zip(drugsmils, drugsmils1)]
drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip()), smiles.strip()) for id, smiles in zip(drug_list, drug_list_smiles)]
#print(drug_id_mol_graph_tup )
drug_id_mol_graph_tup1 = [(id, smiles) for id, smiles in zip(drug_list, drug_list_smiles)]

# Gettings information and features of atoms
ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])
AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
max_valence = max(max_valence, 9)
AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0
print(999999999999)
print(ATOM_MAX_NUM)
drug_id_mol_tup = []
# symbols = list()
# drug_smile_dict = {}
# for mol in drug_id_mol_graph_tup:
#             symbols.extend(atom.GetSymbol() for atom in mol[1].GetAtoms())
#
# symbols = list(set(symbols))
a=np.load('symbo.npy')
symbols=a.tolist()
MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol,smiles,symbols)
                                for drug_id, mol,smiles in drug_id_mol_graph_tup}
MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}
#print(MOL_EDGE_LIST_FEAT_MTX.shape)
TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])
import pickle

newMOL_EDGE_LIST_FEAT_MTX = {drug_list.index(drug_id): get_mol_edge_list_and_feat_mtx(mol,smiles,symbols)
                                for drug_id, mol,smiles in drug_id_mol_graph_tup}
newMOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in newMOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}
# with open("alldrugdata.pkl",'wb') as f:
#
#     pickle.dump(newMOL_EDGE_LIST_FEAT_MTX,f,pickle.HIGHEST_PROTOCOL)
#
# with open("123.pkl",'rb') as f:
#     dic = pickle.load(f)

##### DDI statistics and counting #######
df_all_pos_ddi = pd.read_csv('data/newddi.csv')
all_pos_tup = [(h, t, r) for h, t, r in zip(df_all_pos_ddi['d1'], df_all_pos_ddi['d2'], df_all_pos_ddi['type'])]


ALL_DRUG_IDS, _,_ = zip(*drug_id_mol_graph_tup)
ALL_DRUG_IDS = np.array(list(set(ALL_DRUG_IDS)))
ALL_TRUE_H_WITH_TR = defaultdict(list)
ALL_TRUE_T_WITH_HR = defaultdict(list)
# I won't corrupt the relations
# self.true_r_with_ht = defaultdict(list)
FREQ_REL = defaultdict(int)
ALL_H_WITH_R = defaultdict(dict)
ALL_T_WITH_R = defaultdict(dict)
ALL_TAIL_PER_HEAD = {}
ALL_HEAD_PER_TAIL = {}

for h, t, r in all_pos_tup:
    ALL_TRUE_H_WITH_TR[(t, r)].append(h)
    ALL_TRUE_T_WITH_HR[(h, r)].append(t)
    FREQ_REL[r] += 1.0
    ALL_H_WITH_R[r][h] = 1
    ALL_T_WITH_R[r][t] = 1

for t, r in ALL_TRUE_H_WITH_TR:
    ALL_TRUE_H_WITH_TR[(t, r)] = np.array(list(set(ALL_TRUE_H_WITH_TR[(t, r)])))
for h, r in ALL_TRUE_T_WITH_HR:
    ALL_TRUE_T_WITH_HR[(h, r)] = np.array(list(set(ALL_TRUE_T_WITH_HR[(h, r)])))

for r in FREQ_REL:
    ALL_H_WITH_R[r] = np.array(list(ALL_H_WITH_R[r].keys()))
    ALL_T_WITH_R[r] = np.array(list(ALL_T_WITH_R[r].keys()))
    ALL_HEAD_PER_TAIL[r] = FREQ_REL[r] / len(ALL_T_WITH_R[r])
    ALL_TAIL_PER_HEAD[r] = FREQ_REL[r] / len(ALL_H_WITH_R[r])


class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)



def create_graph_data(id):
    edge_index = MOL_EDGE_LIST_FEAT_MTX[id][0]
    features = MOL_EDGE_LIST_FEAT_MTX[id][1]
    edge_attr1=MOL_EDGE_LIST_FEAT_MTX[id][2]

    return Data(x=features, edge_index=edge_index, edge_attr=edge_attr1, ids=id)

def fanhui2d(idx):
    # drug_list = []
    # with open('data/drug_list.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         if row[0] != 'drugbank_id':
    #             drug_list.append(row[0])
    pos_h_samples1 = []

    ids23d = idx

    druglistbatch=[]
    for i in range(0,ids23d.shape[0]//2):
        druglistbatch.append(drug_list[ids23d[i]])
    for h in druglistbatch:

        h_data1 = create_graph_data(h)
        pos_h_samples1.append(h_data1)
    pos_h_samples1 = Batch.from_data_list(pos_h_samples1)
    pos_tri2d = pos_h_samples1

    druglistbatch = []
    pos_h_samples1 = []
    for i in range(ids23d.shape[0]//2,ids23d.shape[0]):
        druglistbatch.append(drug_list[ids23d[i]])
    for h in druglistbatch:
        h_data1 = create_graph_data(h)
        pos_h_samples1.append(h_data1)
    pos_h_samples1 = Batch.from_data_list(pos_h_samples1)

    pos_tri2dt = pos_h_samples1
    return pos_tri2d,pos_tri2dt

def fanhui2d_one(idx):
    # drug_list = []
    # with open('data/drug_list.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         if row[0] != 'drugbank_id':
    #             drug_list.append(row[0])
    pos_h_samples1 = []

    ids23d = idx
    drug=drug_list[ids23d]
    drug_data = create_graph_data(drug)
    #     pos_h_samples1.append(h_data1)
    # pos_h_samples1 = Batch.from_data_list(pos_h_samples1)
    # pos_tri2d = pos_h_samples1
    #
    # druglistbatch = []
    # pos_h_samples1 = []
    # for i in range(ids23d.shape[0]//2,ids23d.shape[0]):
    #     druglistbatch.append(drug_list[ids23d[i]])
    # for h in druglistbatch:
    #     h_data1 = create_graph_data(h)
    #     pos_h_samples1.append(h_data1)
    # pos_h_samples1 = Batch.from_data_list(pos_h_samples1)
    #
    # pos_tri2dt = pos_h_samples1
    return drug_data#pos_tri2d,pos_tri2dt
def fanhui2dfrommol(mol,smiles):
    atom_symbols = np.load('symbo.npy')
    atom_symbols = atom_symbols.tolist()
    mat=get_mol_edge_list_and_feat_mtx(mol, smiles, atom_symbols)
    edge_index = mat[0]
    features = mat[1]
    edge_attr1 = mat[2]

    return Data(x=features, edge_index=edge_index, edge_attr=edge_attr1)
    # # drug_list = []
    # # with open('data/drug_list.csv', 'r') as f:
    # #     reader = csv.reader(f)
    # #     for row in reader:
    # #         if row[0] != 'drugbank_id':
    # #             drug_list.append(row[0])
    # pos_h_samples1 = []
    #
    # ids23d = idx
    # drug=drug_list[ids23d]
    # drug_data = create_graph_data(drug)
    #
    # return drug_data#pos_tri2d,pos_tri2dt



