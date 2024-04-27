import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
#device = torch.device('cpu')
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GATConv, GINConv

import numpy as np
import csv
import random

from torch.nn.modules.container import ModuleList

# from torch_geometric.nn import (GATConv,
#                                 SAGPooling,
#                                 LayerNorm,
#                                 global_mean_pool,
#                                 max_pool_neighbor_x,
#                                 global_add_pool)

from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear, GRU, Parameter
from torch.nn.functional import leaky_relu
from torch_geometric.nn import Set2Set, NNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import GPSConv,GINEConv
from torch_geometric.utils import softmax
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn import global_mean_pool




import os.path as osp
import warnings
from math import pi as PI
from typing import Callable, Dict, Optional, Tuple


from torch import Tensor
from torch.nn import Embedding, ModuleList, Sequential


from torch_geometric.nn.models import  DimeNet#dimenet
from torch_geometric.nn import DimeNet, SchNet #,DimeNetPlusPlus
import sympy as sym

#from egnn_pytorch import EGNN,EGNN_Sparse_Network,EGNN_Sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (MessagePassing, global_add_pool, global_max_pool, global_mean_pool)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

#from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver,normalization_resolver

from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
import inspect
from multiheadattention import MultiheadAttention
#from typing import Any, Dict, Optional
# from torch import nn, einsum, broadcast_tensors
#
#
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
#
# # types
#
# from typing import Optional, List, Union
#
# import torch
# from torch import nn, einsum, broadcast_tensors
# import torch.nn.functional as F
#
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
#
# # types
#
# from typing import Optional, List, Union
#
# # pytorch geometric
#
# try:
#     import torch_geometric
#     from torch_geometric.nn import MessagePassing
#     from torch_geometric.typing import Adj, Size, OptTensor, Tensor
# except:
#     Tensor = OptTensor = Adj = MessagePassing = Size = object
#     PYG_AVAILABLE = False
#
#     # to stop throwing errors from type suggestions
#     Adj = object
#     Size = object
#     OptTensor = object
#     Tensor = object
#
# from egnn_pytorch1 import *
#pytorch geometric

try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.typing import Adj, Size, OptTensor, Tensor
except:
    Tensor = OptTensor = Adj = MessagePassing = Size = object
    PYG_AVAILABLE = False

    # to stop throwing errors from type suggestions
    Adj = object
    Size = object
    OptTensor = object
    Tensor = object

#from egnn_pytorch1 import *
#from transformers import AutoTokenizer, BertModel,BioGptModel,AutoModel,BertTokenizer, BertModel
import torch_geometric.transforms as T


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

        # Global attention transformer-style model.
        #print(111111111111)
        #print(x.shape)

        h, mask = to_dense_batch(x, batch)
        #h=x


        #print(mask.dtype)
        #mask=mask.byte()
        #mask=mask.bool()
        #mask=mask.to(dtype=torch.bool)
        #print(mask)
        #print(~mask)
        # if isinstance(self.attn, torch.nn.MultiheadAttention):
        #     h, _ = self.attn(h, h, h, key_padding_mask=~mask,
        #                      need_weights=False) need_weights=True. If average_attn_weights=True
        #if isinstance(self.attn, torch.nn.MultiheadAttention):
        h, attweight = self.attn(h, h, h,key_padding_mask=~mask,#,attn_mask=~mask,
                             need_weights=True,average_attn_weights=True)
        #print(attweight.shape)
            #print(h.shape)
            #print(attweight.shape)
        # elif isinstance(self.attn, PerformerAttention):
        #     h = self.attn(h, mask=mask)
        #print(h.shape)
        #print(211111111111111111)#
        #print(h.shape)
        h = h[mask]
        #attweight = torch.mean(attweight1, dim=1, keepdim=True)
        #print(attweight)
        #print(attweight.shape)
        attweight=attweight[mask]
        #print(attweight)
        #print(attweight.shape)
        #print(h.shape)
        #print(mask.shape)
        #mask=mask.view(-1)
        #attweight = attweight[mask]
        #attweight1=attweight.cpu().detach().numpy()
        #attweight1=np.sum(attweight1,1)
        #attweight = torch.sum(attweight, dim=1, keepdim=True)
        #print(attweight)
        #print(attweight1)
        #attweight=F.softmax(attweight,dim=1)


        #print(attweight.shape)
        #print(attweight)
        #print(h)
        #print(attweight.shape)
        #print(attweight[0:2])
        #attweight=torch.mean(attweight,dim=1,keepdim=True)
        #print(attweight[0:2])
        #attweight = torch.mean(attweight, dim=1, keepdim=True)
        #print(attweight)
        #print(attweight.shape)
        #print(h.shape)
        #print(attweight.shape)
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


class MMDDI2d_256(nn.Module):
    def __init__(self,  dropout,args,num_tasks):
        super(MMDDI2d_256, self).__init__()
        channels = 128
        num_layers = 3

        # self.edge_emb = Embedding(6, channels)
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        # self.node_emb = Embedding(457, channels - 8)
        self.encoder2D = ModuleList()
        attn_kwargs = {'dropout': args.mattdropout}
        # self.convs = ModuleList()
        for _ in range(num_layers):
            nn1 = Sequential(
                Linear(channels, channels),
                nn.ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv1(channels, GINEConv(nn1), heads=4, attn_type='multihead', attn_kwargs=attn_kwargs,
                            dropout=args.mdropout)
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
        self.args=args
        self.attnfea = torch.nn.MultiheadAttention(128, 4, batch_first=True,dropout=args.mattdropout)
        self.mlp1 = nn.ModuleList([nn.Linear(128, num_tasks)])
        self.dropout = dropout

    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp1[i](vectors)

        return vectors

    def forward(self, alltwograph):

        alltwograph = self.transform(alltwograph)

        h_data_2d = alltwograph  # [idx[1]]
        x_pe = self.pe_norm(h_data_2d.pe)

        x = torch.cat((self.linx(h_data_2d.x.float()), self.pe_lin(x_pe)), 1)
        edge_attr = self.linedgea(h_data_2d.edge_attr.float())  # self.edge_emb(h_data_2d.edge_attr)

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

        # original
        zl = z1 + zl
        zl = torch.squeeze(zl)
        zall= self.MLP(zl, 1)
        return zall

    def forward_view(self, twograph):

        h_data_2d = self.transform(twograph)#[idx[1]]
        x_pe = self.pe_norm(h_data_2d.pe)
        x = torch.cat((self.linx(h_data_2d.x), self.pe_lin(x_pe)), 1)
        edge_attr = self.linedgea(h_data_2d.edge_attr)  # self.edge_emb(h_data_2d.edge_attr)
        attweight1=0
        for conv in self.encoder2D:
            x,attweight = conv(x, h_data_2d.edge_index, edge_attr=edge_attr)
            attweight1=attweight1+attweight
        attweightaver=attweight1/3
        return attweightaver

