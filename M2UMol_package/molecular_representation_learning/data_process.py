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
import copy
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from rdkit.Chem.rdchem import BondType as BT
BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
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
    features = [(atom.GetIdx(), atom_features(atom,atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()  # to make sure that the feature matrix is aligned according to the idx of the atom
    _, features = zip(*features)
    features = torch.stack(features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),*edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (
    torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats
    undirected_edge_list = edge_list

    return undirected_edge_list.T, features, edge_feats

def create_all_graph_data(smiles_list):
    a=np.load('symbo.npy')  #load symbols
    symbols=a.tolist()
    smiles_id_list= list(range(0, len(smiles_list)))
    drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip()), smiles.strip()) for id, smiles in zip(smiles_id_list, smiles_list)]
    MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol,smiles,symbols)
                                for drug_id, mol,smiles in drug_id_mol_graph_tup}
    MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}
    return MOL_EDGE_LIST_FEAT_MTX

def construct_graph(MOL_EDGE_LIST_FEAT_MTX,id):
    edge_index = MOL_EDGE_LIST_FEAT_MTX[id][0]
    features = MOL_EDGE_LIST_FEAT_MTX[id][1]
    edge_attr1=MOL_EDGE_LIST_FEAT_MTX[id][2]

    return Data(x=features, edge_index=edge_index, edge_attr=edge_attr1, ids=id)

def construct_graph_batch(all_graph_data):
    hdata = []
    for i,_ in enumerate(all_graph_data):
        hdata.append(construct_graph(all_graph_data,i))
    batch = Batch.from_data_list(hdata)
    return batch