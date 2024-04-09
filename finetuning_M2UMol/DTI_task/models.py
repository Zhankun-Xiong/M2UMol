import torch.nn as nn
import torch.nn.functional as F
import torch
import math
#from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.nn import GATConv, GINConv
from torch.nn.modules.container import ModuleList
#from convert import from_dgl
#from torch_geometric.utils import from_dgl
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
from torch_geometric.utils import softmax
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn import global_mean_pool
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
from typing import Any, Dict, Optional
from torch.nn import Dropout, Linear, Sequential
from torch_geometric.nn.conv import GPSConv,GINEConv
from torch_geometric.nn.resolver import activation_resolver,normalization_resolver
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
import inspect
from torch_geometric.nn import (MessagePassing, global_add_pool, global_max_pool, global_mean_pool)
import warnings
warnings.filterwarnings("ignore")

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DrugBAN(nn.Module):
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        self.drug_extractor = M2UMol() ##############Define our M2UMol as molecular encoder
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        model_dict = self.drug_extractor.state_dict()
        pretrained_dict = torch.load('pre-trained_M2UMol.pt')  #
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        pretrained_dictname = {k for k, v in pretrained_dict.items()}
        print('check loaded pre-trained layer')
        print(pretrained_dictname)

        model_dict.update(pretrained_dict)
        self.drug_extractor.load_state_dict(model_dict)  #
        print('load pre-trained M2UMol success')

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads), #drug_hidden_feats[-1]
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        v_d=torch.unsqueeze(v_d, dim=1)
        #v_d=F.dropout(v_d, p=0.3, training=self.training)
        f, att = self.bcn(v_d, v_p)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

    def forward_getmolatt(self, bg_d):
        #v_d = self.drug_extractor(bg_d)
        att=self.drug_extractor.forward_view(bg_d)
        embedding=self.drug_extractor(bg_d)

        return att,embedding
    
    def forward_getembedding(self, bg_d):
        #v_d = self.drug_extractor(bg_d)
        #att=self.drug_extractor.forward_view(bg_d)
        embedding=self.drug_extractor.forward(bg_d)

        return embedding

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
class M2UMol(nn.Module):
    def __init__(self):
        super(M2UMol, self).__init__()
        channels = 128
        num_layers = 3

        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        self.encoder2D = ModuleList()
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
    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp1[i](vectors)

        return vectors

    def forward(self, alltwograph):
        alltwograph = self.transform(alltwograph)

        h_data_2d = alltwograph  # [idx[1]]
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
        ztext = self.lintotext(feats2d)
        ztext = self.norm(ztext)
        ztext = torch.unsqueeze(ztext, dim=1)
        zbio = self.lintobio(feats2d)
        zbio = self.norm(zbio)
        zbio = torch.unsqueeze(zbio, dim=1)

        z1 = torch.unsqueeze(z1, dim=1)
        z11 = torch.cat((z3d, ztext, zbio), dim=1)
        zl, _ = self.attnfea(z1, z11, z11)
        zl = z1 + zl
        zl = torch.squeeze(zl)

        zall = zl#torch.cat((zl, zr), dim=1)
        zall=zall.view(-1,128)
        zall=self.normf(zall)
        return zall
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
        return attweightaver

# class MolecularGCN(nn.Module):     the molecular encoder used by DrugBAN
#     def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
#         super(MolecularGCN1, self).__init__()
#         self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
#         if padding:
#             with torch.no_grad():
#                 self.init_transform.weight[-1].fill_(0)
#         self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
#         self.output_feats = hidden_feats[-1]
#
#     def forward(self, batch_graph):
#         node_feats = batch_graph.ndata.pop('h')
#         node_feats = self.init_transform(node_feats)
#         node_feats = self.gnn(batch_graph, node_feats)
#         batch_size = batch_graph.batch_size
#         node_feats = node_feats.view(batch_size, -1, self.output_feats)
#         return node_feats


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
