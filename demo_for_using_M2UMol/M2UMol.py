import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import random
from torch.nn.modules.container import ModuleList
from torch.nn import Linear, Parameter
from torch_geometric.nn import global_mean_pool
import warnings
from math import pi as PI
from typing import Callable, Dict, Optional, Tuple
from torch import Tensor
from torch.nn import Embedding, ModuleList, Sequential
import sympy as sym

import torch_geometric.transforms as T
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn import (MessagePassing, global_add_pool, global_max_pool, global_mean_pool)
import inspect
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch.nn import Dropout, Linear, Sequential
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver,normalization_resolver
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from torch.optim import Adam
class GPSConv1(torch.nn.Module):


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
            self.attn = torch.nn.MultiheadAttention(  #MultiheadAttention(
                channels,
                heads,
                batch_first=True,
                **attn_kwargs,
            )
        # elif attn_type == 'performer':
        #     self.attn = PerformerAttention(
        #         channels=channels,
        #         heads=heads,
        #         **attn_kwargs,
        #     )
        else:
            # TODO: Support BigBird
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


        h, mask = to_dense_batch(x, batch)

        h, attweight = self.attn(h, h, h,key_padding_mask=~mask,#,attn_mask=~mask,
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
        #print(attweight)
        return out,attweight


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads}, '
                f'attn_type={self.attn_type})')
class M2UMolencoder(nn.Module):
    def __init__(self,num_layer=3):
        super(M2UMolencoder, self).__init__()
        channels = 128
        num_layers = 3
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        self.encoder2D= ModuleList()
        attn_kwargs = {'dropout': 0.3}
        # self.convs = ModuleList()
        for _ in range(num_layers):
            nn1 = Sequential(
                Linear(channels, channels),
                nn.ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv1(channels, GINEConv(nn1), heads=4, attn_type='multihead', attn_kwargs=attn_kwargs,
                            dropout=0.3)
            self.encoder2D.append(conv)

        self.pe_lin = Linear(20, 8)
        self.pe_norm = nn.BatchNorm1d(20)
        self.depth = 3
        self.dropout = 0.1
        self.norm = nn.LayerNorm(128, 1e-5)
        self.linx = Linear(103, 120)
        self.linedgea = Linear(6, 128)
        self.linto3d = Linear(128, 128)
        self.lintotext = Linear(128, 128)
        self.lintobio = Linear(128, 128)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.attnfea = torch.nn.MultiheadAttention(128, 4, batch_first=True, dropout=0.3)
        self.linfi=Linear(128,128)
        self.normf=nn.BatchNorm1d(128)
        self.classification_mlp = Linear(128, 1)
    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp1[i](vectors)

        return vectors

    def forward(self, alltwograph):

        alltwograph = self.transform(alltwograph)

        h_data_2d = alltwograph 
        x_pe = self.pe_norm(h_data_2d.pe)
        x = torch.cat((self.linx(h_data_2d.x), self.pe_lin(x_pe)), 1)
        edge_attr = self.linedgea(h_data_2d.edge_attr)  # self.edge_emb(h_data_2d.edge_attr)

        for conv in self.encoder2D:
            x, _ = conv(x, h_data_2d.edge_index, h_data_2d.batch, edge_attr=edge_attr)
        feats2d = global_add_pool(x, h_data_2d.batch)

        z1 = self.norm(feats2d)


        z3d = self.linto3d(feats2d)
        z3d = self.norm(z3d)
        z3d = torch.unsqueeze(z3d, dim=1)
        zwenben = self.lintotext(feats2d)
        zwenben = self.norm(zwenben)
        zwenben = torch.unsqueeze(zwenben, dim=1)
        zfinger = self.lintobio(feats2d)
        zfinger = self.norm(zfinger)
        zfinger = torch.unsqueeze(zfinger, dim=1)

        z1 = torch.unsqueeze(z1, dim=1)
        z11 = torch.cat((z3d, zwenben, zfinger), dim=1)
        zl, _ = self.attnfea(z1, z11, z11)
        zl = z1 + zl
        #zl=torch.cat((z1, zl), dim=-1)#
        zl = torch.squeeze(zl)

        #zl=self.linfi(zl)
        zall = zl#torch.cat((zl, zr), dim=1)
        zall=zall.view(-1,128)
        zall=self.normf(zall)
        return zall,z3d,zwenben,zfinger
    def forward_view(self, twograph):

        ######################################################### 2d

        h_data_2d = self.transform(twograph)#[idx[1]]
        x_pe = self.pe_norm(h_data_2d.pe)
        x = torch.cat((self.linx(h_data_2d.x), self.pe_lin(x_pe)), 1)
        edge_attr = self.linedgea(h_data_2d.edge_attr)  # self.edge_emb(h_data_2d.edge_attr)
        attweight1=0
        for conv in self.encoder2D:
            x,attweight = conv(x, h_data_2d.edge_index, edge_attr=edge_attr)
            attweight1=attweight1+attweight

        attweightaver=attweight1/3
        #feats2d = global_add_pool(x, h_data_2d.batch)
        return attweightaver

class M2UMolencoder_withmlp(nn.Module):
    def __init__(self,num_layer=3):
        super(M2UMolencoder_withmlp, self).__init__()
        channels = 128
        num_layers = 3
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        self.encoder2D= ModuleList()
        attn_kwargs = {'dropout': 0.3}
        # self.convs = ModuleList()
        for _ in range(num_layers):
            nn1 = Sequential(
                Linear(channels, channels),
                nn.ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv1(channels, GINEConv(nn1), heads=4, attn_type='multihead', attn_kwargs=attn_kwargs,
                            dropout=0.3)
            self.encoder2D.append(conv)

        self.pe_lin = Linear(20, 8)
        self.pe_norm = nn.BatchNorm1d(20)
        self.depth = 3
        self.dropout = 0.1
        self.norm = nn.LayerNorm(128, 1e-5)
        self.linx = Linear(103, 120)
        self.linedgea = Linear(6, 128)
        self.linto3d = Linear(128, 128)
        self.lintotext = Linear(128, 128)
        self.lintobio = Linear(128, 128)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.attnfea = torch.nn.MultiheadAttention(128, 4, batch_first=True, dropout=0.3)
        self.linfi=Linear(128,128)
        self.normf=nn.BatchNorm1d(128)
        self.classification_mlp = Linear(128, 1)
    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp1[i](vectors)

        return vectors

    def forward(self, alltwograph):

        alltwograph = self.transform(alltwograph)

        h_data_2d = alltwograph
        x_pe = self.pe_norm(h_data_2d.pe)
        x = torch.cat((self.linx(h_data_2d.x), self.pe_lin(x_pe)), 1)
        edge_attr = self.linedgea(h_data_2d.edge_attr)  # self.edge_emb(h_data_2d.edge_attr)

        for conv in self.encoder2D:
            x, _ = conv(x, h_data_2d.edge_index, h_data_2d.batch, edge_attr=edge_attr)
        feats2d = global_add_pool(x, h_data_2d.batch)

        z1 = self.norm(feats2d)


        z3d = self.linto3d(feats2d)
        z3d = self.norm(z3d)
        z3d = torch.unsqueeze(z3d, dim=1)
        zwenben = self.lintotext(feats2d)
        zwenben = self.norm(zwenben)
        zwenben = torch.unsqueeze(zwenben, dim=1)
        zfinger = self.lintobio(feats2d)
        zfinger = self.norm(zfinger)
        zfinger = torch.unsqueeze(zfinger, dim=1)

        z1 = torch.unsqueeze(z1, dim=1)
        z11 = torch.cat((z3d, zwenben, zfinger), dim=1)
        zl, _ = self.attnfea(z1, z11, z11)
        zl = z1 + zl
        #zl=torch.cat((z1, zl), dim=-1)#
        zl = torch.squeeze(zl)

        #zl=self.linfi(zl)
        zall = zl#torch.cat((zl, zr), dim=1)
        zall=zall.view(-1,128)
        zall=self.normf(zall)
        zpre=self.classification_mlp(zall)
        return zpre
    def forward_view(self, twograph):

        ######################################################### 2d

        h_data_2d = self.transform(twograph)#[idx[1]]
        x_pe = self.pe_norm(h_data_2d.pe)
        x = torch.cat((self.linx(h_data_2d.x), self.pe_lin(x_pe)), 1)
        edge_attr = self.linedgea(h_data_2d.edge_attr)  # self.edge_emb(h_data_2d.edge_attr)
        attweight1=0
        for conv in self.encoder2D:
            x,attweight = conv(x, h_data_2d.edge_index, edge_attr=edge_attr)
            attweight1=attweight1+attweight

        attweightaver=attweight1/3
        #feats2d = global_add_pool(x, h_data_2d.batch)
        return attweightaver
def model_load(model,path_pretrained_model):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path_pretrained_model)  #
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dictname = {k for k, v in pretrained_dict.items()}
    print('check loaded pre-trained layer')
    print(pretrained_dictname)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def create_model(lr=1e-3,lr_ft=1e-4,weight_decay=5e-5):

    model = M2UMolencoder_withmlp().cuda()
    model1=torch.compile(model)

    ignored_params1 = list(map(id, model1.classification_mlp.parameters()))
    ignored_params2 = list(map(id, model1.attnfea.parameters()))
    ignored_params = ignored_params1 + ignored_params2 #+ ignored_params3
    base_params1 = filter(lambda p: id(p) not in ignored_params, model1.parameters())
    optimizer = Adam([
        {'params': model1.classification_mlp.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': model1.attnfea.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        {'params': base_params1, 'lr': lr_ft, 'weight_decay': weight_decay},
    ], lr=lr, weight_decay=weight_decay)
    return model, optimizer
