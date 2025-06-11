import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,RGCNConv,GINConv,GATConv
import numpy as np
import csv
import random
from torch.nn.modules.container import ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear, GRU, Parameter
import os.path as osp
import warnings
from math import pi as PI
from typing import Callable, Dict, Optional, Tuple
from torch import Tensor
from torch.nn import Embedding, ModuleList, Sequential
import sympy as sym
try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
except:
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    PYG_AVAILABLE = False
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

from transformers import AutoTokenizer, BertModel,BioGptModel,AutoModel,BertTokenizer, BertModel
import torch_geometric.transforms as T
from torch_geometric.nn.conv import GPSConv,GINEConv
from torch_geometric.nn import (MessagePassing, global_add_pool, global_max_pool, global_mean_pool)
from dig.threedgraph.method import SphereNet,ComENet
import inspect
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver,normalization_resolver
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
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
def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

class GPSConv1(torch.nn.Module):
    ###############2D encoder

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        act: str = 'relu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = 'multihead',
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type

        attn_kwargs = attn_kwargs or {}
        if attn_type == 'multihead':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )

        else:
            raise ValueError(f'{attn_type} is not supported')

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()


    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        **kwargs,) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        h, mask = to_dense_batch(x, batch)

        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, attweight = self.attn(h, h, h, key_padding_mask=~mask,
                             need_weights=True,average_attn_weights=True)

        h = h[mask]
        attweight=attweight[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out,attweight


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads}, '
                f'attn_type={self.attn_type})')

class M2UMol(nn.Module):
    def __init__(self,num_layers=3,hidden1=128,encoder_dropout=0.5,dropout=0.1):
        super(M2UMol, self).__init__()
        channels = hidden1
        num_layers = num_layers
        pe_dim=8
        self.edge_emb = Embedding(6, channels)
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        self.encoder2D = ModuleList()  #GraphGPS
        attn_kwargs = {'dropout': encoder_dropout}
        for _ in range(num_layers):
            nn1 = Sequential(
                Linear(channels, channels),
                nn.ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv1(channels, GINEConv(nn1), heads=4,attn_type='multihead',attn_kwargs=attn_kwargs,dropout=dropout)
            self.encoder2D.append(conv)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)
        self.depth = 3

        self.norm = nn.LayerNorm(128, 1e-5)
        #self.linx = Linear(103, 120)
        #self.linedgea = Linear(6, 128)
        self.linx=AtomEncoder(emb_dim=120)
        self.linedgea=BondEncoder(emb_dim = 128)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)


        self.encoder3D = ComENet(out_channels=128)#ComENet

        self.norm = nn.LayerNorm(128,1e-5)

        #encoders for bio features
        self.lintar = Linear(4463, 128)
        self.linenz = Linear(419, 128)
        self.lincate = Linear(3607, 128)

        self.lintext=Linear(768,128)
        self.linfin=Linear(128,128)
            
        #multi-to-uni adapter
        self.linto3d = Linear(128, 128)
        self.lintotext = Linear(128, 128)
        self.lintobio = Linear(128, 128)

        self.modelcls = Linear(128, 3)


        self.tokenizer = AutoTokenizer.from_pretrained("pretrained-PubMedBERT")
        self.encoderText = AutoModel.from_pretrained("pretrained-PubMedBERT")


    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
    def forward(self,twograph,threegraph,text2d,text,bio):


            ########################## 2D modality
            h_data_2d = self.transform(twograph)
            x_pe = self.pe_norm(h_data_2d.pe)
            x = torch.cat((self.linx(h_data_2d.x), self.pe_lin(x_pe)), 1)
            edge_attr = self.linedgea(h_data_2d.edge_attr)
            for conv in self.encoder2D :
                x,attweight = conv(x, h_data_2d.edge_index, h_data_2d.batch, edge_attr=edge_attr)
            feats2d = global_add_pool(x, h_data_2d.batch)

            ########################## 3D modality
            h_data_3d = threegraph
            feats3d=self.encoder3D(h_data_3d)
            z1=feats2d
            z2=feats3d
            z1 = self.linto3d(z1)
            z1 = self.norm(z1)#generated 3D representation
            z2 = self.norm(z2)

            ########################## Text modality
            h_data_wb = text2d
            h_data_wb = self.transform(h_data_wb)
            x_pe = self.pe_norm(h_data_wb.pe)
            x = torch.cat((self.linx(h_data_wb.x), self.pe_lin(x_pe)), 1)
            edge_attr = self.linedgea(h_data_wb.edge_attr)
            for conv in self.encoder2D:
                x, _ = conv(x, h_data_wb.edge_index, h_data_wb.batch, edge_attr=edge_attr)
            feats_wb = global_add_pool(x, h_data_wb.batch)

            self.drugsum = text
            a = self.tokenizer(self.drugsum, return_tensors="pt", padding=True, truncation=True, max_length=512)
            b = a['input_ids'].to(device)
            c = a['attention_mask'].to(device)
            inputssa, inputssb = b, c
            outputs = self.encoderText(inputssa, inputssb)
            outputs1 = outputs.pooler_output
            textz = outputs1
            textz = self.lintext(textz)
            ztext = feats_wb
            ztext = self.lintotext(ztext)
            ztext = self.norm(ztext) #generated Text representation
            textz = self.norm(textz)

            ########################## Bio modality
            h_data_tz = bio[0].to(device)
            h_data_tz = self.transform(h_data_tz)
            x_pe = self.pe_norm(h_data_tz.pe)
            x = torch.cat((self.linx(h_data_tz.x), self.pe_lin(x_pe)), 1)
            edge_attr = self.linedgea(h_data_tz.edge_attr)
            for conv in self.encoder2D:
                x, attweight = conv(x, h_data_tz.edge_index, h_data_tz.batch, edge_attr=edge_attr)
            feats_tz = global_add_pool(x, h_data_tz.batch)
            targetfea=bio[1].to(device)
            enzymefea=bio[2].to(device)
            drugcatefea=bio[3].to(device)
            targetfea=torch.tensor(targetfea, dtype=torch.float,requires_grad=False)
            enzymefea =torch.tensor(enzymefea, dtype=torch.float,requires_grad=False)
            drugcatefea =torch.tensor(drugcatefea, dtype=torch.float,requires_grad=False)
            tarfea = self.lintar(targetfea)
            enzfea = self.linenz(enzymefea)
            drugcatefea = self.lincate(drugcatefea)
            biofea = (tarfea + enzfea + drugcatefea) / 3
            biofea=self.linfin(biofea)
            zbio = feats_tz
            zbio= self.lintobio(zbio)
            zbio = self.norm(zbio)# generated Bio representation
            biofea= self.norm(biofea)

            ##########################  modality classification
            z11=self.modelcls(z1)
            z22 = self.modelcls(ztext)
            z33 = self.modelcls(zbio)

            return z1, z2, ztext, textz, zbio, biofea, z11, z22, z33

    def forward_view(self, twograph):

        h_data_2d = self.transform(twograph)
        x_pe = self.pe_norm(h_data_2d.pe)
        x = torch.cat((self.linx(h_data_2d.x), self.pe_lin(x_pe)), 1)
        edge_attr = self.linedgea(h_data_2d.edge_attr)
        for conv in self.encoder2D:
            x,attweight = conv(x, h_data_2d.edge_index, h_data_2d.batch, edge_attr=edge_attr)
            attweight1=attweight1+attweight
        attweightaver=attweight1/3



        return attweight1
