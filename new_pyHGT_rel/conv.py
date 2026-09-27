import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv, RGCNConv, SAGEConv, GCN2Conv, GATv2Conv, TransformerConv, FAConv, SuperGATConv, HEATConv, EGConv, FiLMConv, HGTConv
from torch_geometric.nn import GINConv, FusedGATConv, GPSConv, SSGConv, MLP, RGATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
from ordered_set import OrderedSet
import math
from tqdm import tqdm
import numba as nb
import numpy as np
import time
import gc
from torch.nn.init import xavier_normal_, xavier_uniform_
from .helper import *
from .compgcn_message_passing import MessagePassingSelf
from typing import Union, Tuple, Optional
from torch_geometric.typing import Adj, Size, OptTensor, PairTensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from math import log
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul, masked_select_nnz
from torch.nn import Parameter as Param
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):
    pass

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):
    pass

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False, add_self_loops=True, dtype=None):
    fill_value = 2.0 if improved else 1.0
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0, dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.0)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight
        row, col = (edge_index[0], edge_index[1])
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return (edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col])

class Origin_GCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, improved: bool=False, cached: bool=False, add_self_loops: bool=True, normalize: bool=True, bias: bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(Origin_GCNConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor=None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = (cache[0], cache[1])
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    pass

@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    pass

def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')

class RGCNv2Conv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, num_base: Optional[int]=5, num_blocks: Optional[int]=None, negative_slope: float=0.2, aggr: str='mean', root_weight: bool=True, bias: bool=True, **kwargs):
        super(RGCNv2Conv, self).__init__(aggr=aggr, node_dim=0, **kwargs)
        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and block-diagonal-decomposition at the same time.')
        in_channels = in_dim
        out_channels = out_dim
        num_relations = num_rel
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_relations = num_rel
        self.num_bases = num_base
        self.num_blocks = num_blocks
        self.heads = num_head
        self.negative_slope = negative_slope
        self.dropout = dropout
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]
        if num_bases is not None:
            self.weight = Parameter(torch.Tensor(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))
        elif num_blocks is not None:
            assert in_channels[0] % num_blocks == 0 and out_channels % num_blocks == 0
            self.weight = Parameter(torch.Tensor(num_relations, num_blocks, in_channels[0] // num_blocks, out_channels // num_blocks))
            self.register_parameter('comp', None)
        else:
            self.weight = Parameter(torch.Tensor(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)
        if root_weight:
            self.root = Param(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)
        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha = None
        self.att = Parameter(torch.Tensor(1, self.heads, self.out_channels // self.heads))
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        self.act = torch.tanh
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(self.out_dim)
        self.w_rel = nn.Parameter(torch.Tensor(in_dim, out_dim)).cuda()
        self.w_node = nn.Parameter(torch.Tensor(in_dim, out_dim)).cuda()
        self.reset_parameters()
        self.reset_parameters2()

    def reset_parameters2(self):
        node_size = self.p.node_num_bases * self.in_dim
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.w_rel.data)
            xavier_normal_(self.w_node.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            xavier_uniform_(self.w_rel.data)
            xavier_uniform_(self.w_node.data)

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj):
        length = node_type.size(1)
        node_maintype = node_type
        k_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        H, C = (self.heads, self.out_channels // self.heads)
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = k_sum
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)
        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]
        size = (x_l.size(0), x_r.size(0))
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(self.num_relations, self.in_channels_l, self.out_channels)
        if self.num_blocks is not None:
            if x_l.dtype == torch.long and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported for non-continuous input features.')
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out += h.contiguous().view(-1, self.out_channels)
        else:
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                if tmp.sum() == 0:
                    continue
                if x_l.dtype == torch.long:
                    out += self.propagate(tmp, x=weight[i, x_l], size=size)
                else:
                    h = self.propagate(edge_index=tmp, x=x_l, size=size, edge_type=edge_type[edge_type == i], rel_embed=rel_embed, node_maintype=node_type, type_count=type_count)
                    h = h.view(-1, H * C)
                    out = out + h @ weight[i]
        r_m = torch.bmm(rel_embed.unsqueeze(1), weight).squeeze(-2)
        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root
        if self.bias is not None:
            out += self.bias
        out = self.drop(self.act(self.bn(out)))
        r_m = self.drop(self.act(r_m))
        return (out, r_m)

    def message(self, x_j, x_i, edge_type, rel_embed, edge_index, node_maintype, type_count, index, ptr, size_i: Optional[int]) -> Tensor:
        H, C = (self.heads, self.out_channels // self.heads)
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        try:
            xj_rel = self.rel_transform(x_j, rel_emb)
        except Exception:
            print('x_j.shape:', x_j.shape)
            print('rel_emb.shape:', rel_emb.shape)
            print('rel_embed.shape:', rel_embed.shape)
            print('edge_type.shape:', edge_type.shape)
            print('edge_index.shape:', edge_index.shape)
            raise Exception('test')
        xj_rel = torch.matmul(xj_rel, self.w_node).view(-1, H, C)
        x = x_i + x_j
        x = x.view(-1, H, C)
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return xj_rel * alpha.unsqueeze(-1)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_relations)

class RGCNv3Conv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, num_base: Optional[int]=5, num_blocks: Optional[int]=None, negative_slope: float=0.2, aggr: str='mean', root_weight: bool=True, bias: bool=False, **kwargs):
        super(RGCNv3Conv, self).__init__(aggr=aggr, node_dim=0, **kwargs)
        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and block-diagonal-decomposition at the same time.')
        in_channels = in_dim
        out_channels = out_dim
        num_relations = num_rel
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_relations = num_rel
        self.num_bases = num_base
        self.num_blocks = num_blocks
        self.heads = num_head
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.num_rels = num_rel - 1
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]
        if num_bases is not None:
            self.weight = Parameter(torch.Tensor(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))
        elif num_blocks is not None:
            assert in_channels[0] % num_blocks == 0 and out_channels % num_blocks == 0
            self.weight = Parameter(torch.Tensor(num_relations, num_blocks, in_channels[0] // num_blocks, out_channels // num_blocks))
            self.register_parameter('comp', None)
        else:
            self.weight = Parameter(torch.Tensor(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)
        if root_weight:
            self.root = Param(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)
        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha = None
        self.att = Parameter(torch.Tensor(1, self.heads, self.out_channels // self.heads))
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        self.act = torch.tanh
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(self.out_dim)
        self.w_rel = nn.Parameter(torch.Tensor(in_dim, out_dim)).cuda()
        self.w_node = nn.Parameter(torch.Tensor(in_dim, out_dim)).cuda()
        self.reset_parameters()
        self.reset_parameters2()

    def reset_parameters2(self):
        node_size = self.p.node_num_bases * self.in_dim
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.w_rel.data)
            xavier_normal_(self.w_node.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            xavier_uniform_(self.w_rel.data)
            xavier_uniform_(self.w_node.data)

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        glorot(self.att)
        zeros(self.bias)

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        return norm

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj):
        num_ent = x.size(0)
        length = node_type.size(1)
        node_maintype = node_type
        k_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        H, C = (self.heads, self.out_channels // self.heads)
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = k_sum
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)
        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]
        size = (x_l.size(0), x_r.size(0))
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(self.num_relations, self.in_channels_l, self.out_channels)
        if self.num_blocks is not None:
            if x_l.dtype == torch.long and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported for non-continuous input features.')
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out += h.contiguous().view(-1, self.out_channels)
        else:
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                if tmp.sum() == 0:
                    continue
                if x_l.dtype == torch.long:
                    out += self.propagate(tmp, x=weight[i, x_l], size=size)
                else:
                    edge_norm = self.compute_norm(tmp, num_ent)
                    h = self.propagate(edge_index=tmp, x=x_l, size=size, edge_type=edge_type[edge_type == i], edge_norm=edge_norm, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count)
                    out = out + h @ weight[i]
        r_m = torch.bmm(rel_embed.unsqueeze(1), weight).squeeze(-2)
        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root
        if self.bias is not None:
            out += self.bias
        out = self.drop(self.act(self.bn(out)))
        r_m = self.drop(self.act(r_m))
        return (out, r_m)

    def message(self, x_j, x_i, edge_type, rel_embed, edge_index, node_maintype, type_count, edge_norm, index, ptr, size_i: Optional[int]) -> Tensor:
        H, C = (self.heads, self.out_channels // self.heads)
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        try:
            xj_rel = self.rel_transform(x_j, rel_emb)
        except Exception:
            print('x_j.shape:', x_j.shape)
            print('rel_emb.shape:', rel_emb.shape)
            print('rel_embed.shape:', rel_embed.shape)
            print('edge_type.shape:', edge_type.shape)
            print('edge_index.shape:', edge_index.shape)
            raise Exception('test')
        xj_rel = torch.matmul(xj_rel, self.w_node)
        return xj_rel * edge_norm.view(-1, 1)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_relations)

class FastRGCNConv(RGCNv2Conv):

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]], edge_index: Adj, edge_type: OptTensor=None):
        self.fuse = False
        assert self.aggr in ['add', 'sum', 'mean']
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)
        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]
        size = (x_l.size(0), x_r.size(0))
        out = self.propagate(edge_index, x=x_l, edge_type=edge_type, size=size)
        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j: Tensor, edge_type: Tensor, index: Tensor) -> Tensor:
        weight = self.weight
        if self.num_bases is not None:
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(self.num_relations, self.in_channels_l, self.out_channels)
        if self.num_blocks is not None:
            if x_j.dtype == torch.long:
                raise ValueError('Block-diagonal decomposition not supported for non-continuous input features.')
            weight = weight[edge_type].view(-1, weight.size(2), weight.size(3))
            x_j = x_j.view(-1, 1, weight.size(1))
            return torch.bmm(x_j, weight).view(-1, self.out_channels)
        else:
            if x_j.dtype == torch.long:
                weight_index = edge_type * weight.size(1) + index
                return weight.view(-1, self.out_channels)[weight_index]
            return torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

    def aggregate(self, inputs: Tensor, edge_type: Tensor, index: Tensor, dim_size: Optional[int]=None) -> Tensor:
        if self.aggr == 'mean':
            norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
            norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
            norm = torch.gather(norm, 1, edge_type.view(-1, 1))
            norm = 1.0 / norm.clamp_(1.0)
            inputs = norm * inputs
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)

class GATv1Conv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int=1, concat: bool=True, negative_slope: float=0.2, dropout: float=0.0, add_self_loops: bool=True, edge_dim: Optional[int]=None, fill_value: Union[float, Tensor, str]='mean', bias: bool=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATv1Conv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False, weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False, weight_initializer='glorot')
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and (not concat):
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor=None, size: Size=None, return_attention_weights=None):
        H, C = (self.heads, self.out_channels)
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)
        x = (x_src, x_dst)
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=self.fill_value, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError("The usage of 'edge_attr' and 'add_self_loops' simultaneously is currently not yet supported for 'edge_index' in a 'SparseTensor' form")
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr, size=size)
        alpha = self._alpha
        assert alpha is not None
        self._alpha = None
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return (out, (edge_index, alpha))
            elif isinstance(edge_index, SparseTensor):
                return (out, edge_index.set_value(alpha, layout='coo'))
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, edge_attr: OptTensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

class GATv3Conv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, concat: bool=True, negative_slope: float=0.2, add_self_loops: bool=False, bias: bool=True, share_weights: bool=False, **kwargs):
        super(GATv3Conv, self).__init__(node_dim=0, aggr='add', **kwargs)
        self.in_channels = in_dim
        self.out_channels = out_dim // num_head
        self.heads = num_head
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.lin_l = Linear(self.in_channels, self.heads * self.out_channels, bias=bias, weight_initializer='glorot')
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(self.in_channels, self.heads * self.out_channels, bias=bias, weight_initializer='glorot')
        self.att = Parameter(torch.Tensor(1, self.heads, self.out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(self.heads * self.out_channels))
        elif bias and (not concat):
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha = None
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        self.act = F.gelu
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(self.out_dim)
        self.w_rel = nn.Parameter(torch.Tensor(in_dim, out_dim)).cuda()
        self.reset_parameters()
        self.reset_parameters2()

    def reset_parameters2(self):
        node_size = self.p.node_num_bases * self.in_dim
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.w_rel.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            xavier_uniform_(self.w_rel.data)

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj, size: Size=None, return_attention_weights: bool=None):
        H, C = (self.heads, self.out_channels)
        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = (x[0], x[1])
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
        assert x_l is not None
        assert x_r is not None
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        out = self.propagate(edge_index, x=(x_l, x_r), size=size, edge_type=edge_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count)
        alpha = self._alpha
        self._alpha = None
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        out = self.bn(out)
        rel_m = torch.matmul(rel_embed, self.w_rel)[:-1]
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return (self.act(out), self.act(rel_m), (edge_index, alpha))
            elif isinstance(edge_index, SparseTensor):
                return (self.act(out), self.act(rel_m), edge_index.set_value(alpha, layout='coo'))
        else:
            return (self.drop(self.act(out)), rel_m)

    def message(self, x_j, x_i, edge_type, rel_embed, edge_index, x, node_maintype, type_count, index, ptr, size_i: Optional[int]) -> Tensor:
        H, C = (self.heads, self.out_channels)
        length = node_maintype.size(1)
        k_sum_j = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x[0].view(-1, H * C).unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum_j = k_sum_j + k_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x[0].view(-1, H * C).unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum_j = k_sum_j + k_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        x_j = torch.index_select(k_sum_j, 0, edge_index.t()[:, 0])
        length = node_maintype.size(1)
        k_sum_i = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x[1].view(-1, H * C).unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum_i = k_sum_i + k_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x[1].view(-1, H * C).unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum_i = k_sum_i + k_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        x_i = torch.index_select(k_sum_i, 0, edge_index.t()[:, 1])
        x = x_i + x_j
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        x = self.rel_transform(x, rel_emb)
        x = x.view(-1, H, C)
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j.view(-1, H, C) * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

class GATv4Conv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, concat: bool=True, negative_slope: float=0.2, add_self_loops: bool=False, bias: bool=True, share_weights: bool=False, **kwargs):
        super(GATv4Conv, self).__init__(node_dim=0, aggr='add', **kwargs)
        self.in_channels = in_dim
        self.out_channels = out_dim // num_head
        self.heads = num_head
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.lin_l = Linear(self.in_channels, self.heads * self.out_channels, bias=bias, weight_initializer='glorot')
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(self.in_channels, self.heads * self.out_channels, bias=bias, weight_initializer='glorot')
        self.att = Parameter(torch.Tensor(1, self.heads, self.out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(self.heads * self.out_channels))
        elif bias and (not concat):
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha = None
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        self.act = torch.tanh
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
                self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(self.out_dim)
        self.w_rel = nn.Parameter(torch.Tensor(in_dim, out_dim)).cuda()
        self.w_node = nn.Parameter(torch.Tensor(in_dim, out_dim)).cuda()
        self.reset_parameters()
        self.reset_parameters2()

    def reset_parameters2(self):
        node_size = self.p.node_num_bases * self.in_dim
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.k_bias.data)
                xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.w_rel.data)
            xavier_normal_(self.w_node.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.k_bias)
                get_uniform(node_size, self.k_bias_att)
            xavier_uniform_(self.w_rel.data)
            xavier_uniform_(self.w_node.data)

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj, size: Size=None, return_attention_weights: bool=None):
        length = node_type.size(1)
        node_maintype = node_type
        k_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        if self.p.bias != 0:
            k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                else:
                    k_b = 0
                k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                else:
                    k_b = 0
                k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        H, C = (self.heads, self.out_channels)
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        out = self.propagate(edge_index, x=k_sum, size=size, edge_type=edge_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count)
        alpha = self._alpha
        self._alpha = None
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        out = self.bn(out)
        rel_m = torch.matmul(rel_embed, self.w_rel)[:-1]
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return (self.act(out), self.act(rel_m), (edge_index, alpha))
            elif isinstance(edge_index, SparseTensor):
                return (self.act(out), self.act(rel_m), edge_index.set_value(alpha, layout='coo'))
        else:
            return (self.drop(self.act(out)), rel_m)

    def message(self, x_j, x_i, edge_type, rel_embed, edge_index, node_maintype, type_count, index, ptr, size_i: Optional[int]) -> Tensor:
        H, C = (self.heads, self.out_channels)
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        xj_rel = torch.matmul(xj_rel, self.w_node).view(-1, H, C)
        x = x_i + x_j
        x = x.view(-1, H, C)
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return xj_rel * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

class HRGATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, concat: bool=True, negative_slope: float=0.2, add_self_loops: bool=False, bias: bool=True, share_weights: bool=False, **kwargs):
        super(HRGATConv, self).__init__(node_dim=0, aggr='add', **kwargs)
        self.in_channels = in_dim
        self.out_channels = out_dim // num_head
        self.heads = num_head
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.lin_l = Linear(self.in_channels, self.heads * self.out_channels, bias=bias, weight_initializer='glorot')
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(self.in_channels, self.heads * self.out_channels, bias=bias, weight_initializer='glorot')
        self.att = Parameter(torch.Tensor(1, self.heads, self.out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(self.heads * self.out_channels))
        elif bias and (not concat):
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha = None
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        self.act = torch.tanh
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        if self.p.re_num_bases > 0:
            self.r_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, in_dim, out_dim)).cuda()
            self.r_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.r_bias = nn.Parameter(torch.Tensor(self.p.re_num_bases, out_dim)).cuda()
            self.r_bias_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(self.out_dim)
        self.w_rel = nn.Parameter(torch.Tensor(in_dim, out_dim)).cuda()
        self.w_node = nn.Parameter(torch.Tensor(in_dim, out_dim)).cuda()
        self.reset_parameters()
        self.reset_parameters2()

    def reset_parameters2(self):
        node_size = self.p.node_num_bases * self.in_dim
        rel_size = self.p.re_num_bases * self.in_dim
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.r_basis.data)
            xavier_normal_(self.r_att.data)
            xavier_normal_(self.r_bias.data)
            xavier_normal_(self.r_bias_att.data)
            xavier_normal_(self.w_rel.data)
            xavier_normal_(self.w_node.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            get_uniform(rel_size, self.k_bias)
            get_uniform(rel_size, self.k_bias_att)
            xavier_uniform_(self.r_basis.data)
            xavier_uniform_(self.r_att.data)
            get_uniform(rel_size, self.r_bias)
            get_uniform(rel_size, self.r_bias_att)
            xavier_uniform_(self.w_rel.data)
            xavier_uniform_(self.w_node.data)

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj, size: Size=None, return_attention_weights: bool=None):
        length = node_type.size(1)
        node_maintype = node_type
        k_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        r_ws = torch.matmul(self.r_att, self.r_basis.view(self.p.re_num_bases, -1))
        r_ws = r_ws.view(self.num_relations, self.in_dim, self.out_dim)
        r_bs = torch.matmul(self.r_bias_att, self.r_bias)
        r_w = r_ws
        r_b = r_bs
        r_m = torch.bmm(rel_embed.unsqueeze(1), r_w).squeeze(-2) + r_b
        H, C = (self.heads, self.out_channels)
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        out = self.propagate(edge_index, x=k_sum, size=size, edge_type=edge_type, rel_embed=r_m, node_maintype=node_type, type_count=type_count)
        alpha = self._alpha
        self._alpha = None
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        out = self.bn(out)
        rel_m = torch.matmul(rel_embed, self.w_rel)[:-1]
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return (self.act(out), self.act(rel_m), (edge_index, alpha))
            elif isinstance(edge_index, SparseTensor):
                return (self.act(out), self.act(rel_m), edge_index.set_value(alpha, layout='coo'))
        else:
            return (self.drop(self.act(out)), rel_m)

    def message(self, x_j, x_i, edge_type, rel_embed, edge_index, x, node_maintype, type_count, index, ptr, size_i: Optional[int]) -> Tensor:
        H, C = (self.heads, self.out_channels)
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        xj_rel = torch.matmul(xj_rel, self.w_node).view(-1, H, C)
        x = x_i + x_j
        x = x.view(-1, H, C)
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return xj_rel * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

class IndRelDenseMTHGCLConv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(IndRelDenseMTHGCLConv, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        self.num_type_bases = 5
        self.norms = nn.ModuleList()
        for t in range(1):
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.v_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.v_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.v_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.v_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.q_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.q_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.q_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.q_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.a_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim, out_dim)).cuda()
            self.a_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.a_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.a_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        self.drop = nn.Dropout(dropout)
        self.hide_linears = nn.ModuleList()
        for t in range(self.p.n_linear):
            self.hide_linears.append(nn.Linear(out_dim * self.p.dimrate_linear, out_dim * self.p.dimrate_linear))
        self.in_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.out_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        if self.p.skip1:
            self.skip1 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.skip2:
            self.skip2 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.re_num_bases > 0:
            self.at_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.at_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.ms_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.ms_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.pr_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads)).cuda()
            self.pr_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.rel_at_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.rel_at_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.rel_ms_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.rel_ms_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.rel_pr_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads)).cuda()
            self.rel_pr_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
        self.reset_parameters()
        self.relation_att = torch.matmul(self.at_att, self.at_basis.view(self.p.re_num_bases, -1))
        self.relation_att = self.relation_att.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
        self.relation_msg = torch.matmul(self.ms_att, self.ms_basis.view(self.p.re_num_bases, -1))
        self.relation_msg = self.relation_msg.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
        self.relation_pri = torch.matmul(self.pr_att, self.pr_basis)
        self.rel_relation_att = torch.matmul(self.rel_at_att, self.rel_at_basis.view(self.p.re_num_bases, -1))
        self.rel_relation_att = self.rel_relation_att.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
        self.rel_relation_msg = torch.matmul(self.rel_ms_att, self.rel_ms_basis.view(self.p.re_num_bases, -1))
        self.rel_relation_msg = self.relation_msg.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
        self.rel_relation_pri = torch.matmul(self.rel_pr_att, self.rel_pr_basis)
        self.rk_linear = nn.Linear(self.in_dim, self.out_dim)
        self.rv_linear = nn.Linear(self.in_dim, self.out_dim)
        self.ra_linear = nn.Linear(self.out_dim, self.out_dim)
        self.rnorm = nn.LayerNorm(out_dim)
        self.rhide_linears = nn.ModuleList()
        for t in range(self.p.n_linear):
            self.rhide_linears.append(nn.Linear(out_dim * self.p.dimrate_linear, out_dim * self.p.dimrate_linear))
        self.rin_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.rout_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.rout_norm = nn.LayerNorm(out_dim)

    def reset_parameters(self):
        node_size = self.p.node_num_bases * self.in_dim
        re_size = self.p.re_num_bases * self.n_heads
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.q_basis.data)
            xavier_normal_(self.q_att.data)
            xavier_normal_(self.q_bias.data)
            xavier_normal_(self.q_bias_att.data)
            xavier_normal_(self.v_basis.data)
            xavier_normal_(self.v_att.data)
            xavier_normal_(self.v_bias.data)
            xavier_normal_(self.v_bias_att.data)
            xavier_normal_(self.a_basis.data)
            xavier_normal_(self.a_att.data)
            xavier_normal_(self.a_bias.data)
            xavier_normal_(self.a_bias_att.data)
            xavier_normal_(self.at_basis.data)
            xavier_normal_(self.at_att.data)
            xavier_normal_(self.ms_basis.data)
            xavier_normal_(self.ms_att.data)
            xavier_normal_(self.pr_basis.data)
            xavier_normal_(self.pr_att.data)
            xavier_normal_(self.rel_at_basis.data)
            xavier_normal_(self.rel_at_att.data)
            xavier_normal_(self.rel_pr_basis.data)
            xavier_normal_(self.rel_pr_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            xavier_uniform_(self.q_basis.data)
            xavier_uniform_(self.q_att.data)
            get_uniform(node_size, self.q_bias)
            get_uniform(node_size, self.q_bias_att)
            xavier_uniform_(self.v_basis.data)
            xavier_uniform_(self.v_att.data)
            get_uniform(node_size, self.v_bias)
            get_uniform(node_size, self.v_bias_att)
            xavier_uniform_(self.a_basis.data)
            xavier_uniform_(self.a_att.data)
            get_uniform(node_size, self.a_bias)
            get_uniform(node_size, self.a_bias_att)
            xavier_uniform_(self.at_basis.data)
            xavier_uniform_(self.at_att.data)
            xavier_uniform_(self.ms_basis.data)
            xavier_uniform_(self.ms_att.data)
            xavier_uniform_(self.pr_basis.data)
            xavier_uniform_(self.pr_att.data)
            xavier_uniform_(self.rel_at_basis.data)
            xavier_uniform_(self.rel_at_att.data)
            xavier_uniform_(self.rel_ms_basis.data)
            xavier_uniform_(self.rel_ms_att.data)
            xavier_uniform_(self.rel_pr_basis.data)
            xavier_uniform_(self.rel_pr_att.data)
        elif self.p.init_func == 'get_uniform':
            get_uniform(node_size, self.k_basis)
            get_uniform(node_size, self.k_att)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            get_uniform(node_size, self.q_basis)
            get_uniform(node_size, self.q_att)
            get_uniform(node_size, self.q_bias)
            get_uniform(node_size, self.q_bias_att)
            get_uniform(node_size, self.v_basis)
            get_uniform(node_size, self.v_att)
            get_uniform(node_size, self.v_bias)
            get_uniform(node_size, self.v_bias_att)
            get_uniform(node_size, self.a_basis)
            get_uniform(node_size, self.a_att)
            get_uniform(node_size, self.a_bias)
            get_uniform(node_size, self.a_bias_att)
            get_uniform(re_size, self.at_basis)
            get_uniform(re_size, self.at_att)
            get_uniform(re_size, self.ms_basis)
            get_uniform(re_size, self.ms_att)
            get_uniform(re_size, self.pr_basis)
            get_uniform(re_size, self.pr_att)
            get_uniform(re_size, self.rel_at_basis.data)
            get_uniform(re_size, self.rel_at_att.data)
            get_uniform(re_size, self.rel_pr_basis.data)
            get_uniform(re_size, self.rel_pr_att.data)
        elif self.p.init_func == 'get_normal':
            get_normal(self.k_basis)
            get_normal(self.k_att)
            get_normal(self.k_bias)
            get_normal(self.k_bias_att)
            get_normal(self.q_basis)
            get_normal(self.q_att)
            get_normal(self.q_bias)
            get_normal(self.q_bias_att)
            get_normal(self.v_basis)
            get_normal(self.v_att)
            get_normal(self.v_bias)
            get_normal(self.v_bias_att)
            get_normal(self.a_basis)
            get_normal(self.a_att)
            get_normal(self.a_bias)
            get_normal(self.a_bias_att)
            get_normal(self.at_basis)
            get_normal(self.at_att)
            get_normal(self.ms_basis)
            get_normal(self.ms_att)
            get_normal(self.pr_basis)
            get_normal(self.pr_att)
            get_normal(self.rel_at_basis.data)
            get_normal(self.rel_at_att.data)
            get_normal(self.rel_pr_basis.data)
            get_normal(self.rel_pr_att.data)
        else:
            raise Exception('input one of xavier_normal/xavier_uniform/get_uniform/get_normal')

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, rel_inp, adj):
        meta_xs = self.propagate(edge_index, node_inp=node_inp, node_maintype=node_maintype, type_count=type_count, edge_type=edge_type, rel_inp=rel_inp)
        meta_rel = self.rel_process(rel_inp, edge_index)
        return (meta_xs, meta_rel)

    def message(self, edge_index_i, edge_index, edge_type, node_inp, node_maintype, type_count):
        start = time.time()
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp.device)
        length = node_maintype.size(1)
        k_sum = 0
        v_sum = 0
        q_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        v_ws = torch.matmul(self.v_att, self.v_basis.view(self.p.node_num_bases, -1))
        v_ws = v_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        v_bs = torch.matmul(self.v_bias_att, self.v_bias)
        q_ws = torch.matmul(self.q_att, self.q_basis.view(self.p.node_num_bases, -1))
        q_ws = q_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        q_bs = torch.matmul(self.q_bias_att, self.q_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
                v_w = torch.index_select(v_ws, 0, node_maintype[:, i])
                v_b = torch.index_select(v_bs, 0, node_maintype[:, i])
                v_m = torch.bmm(node_inp.unsqueeze(1), v_w).squeeze(-2) + v_b
                v_sum = v_sum + v_m * type_count[:, i].view(-1, 1)
                q_w = torch.index_select(q_ws, 0, node_maintype[:, i])
                q_b = torch.index_select(q_bs, 0, node_maintype[:, i])
                q_m = torch.bmm(node_inp.unsqueeze(1), q_w).squeeze(-2) + q_b
                q_sum = q_sum + q_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
                v_w = torch.index_select(v_ws, 0, node_maintype[:, i])
                v_b = torch.index_select(v_bs, 0, node_maintype[:, i])
                v_m = torch.bmm(node_inp.unsqueeze(1), v_w).squeeze(-2) + v_b
                v_sum = v_sum + v_m * type_count[:, i].view(-1, 1)
                q_w = torch.index_select(q_ws, 0, node_maintype[:, i])
                q_b = torch.index_select(q_bs, 0, node_maintype[:, i])
                q_m = torch.bmm(node_inp.unsqueeze(1), q_w).squeeze(-2) + q_b
                q_sum = q_sum + q_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        k_m = torch.index_select(k_sum, 0, edge_index.t()[:, 1])
        v_m = torch.index_select(v_sum, 0, edge_index.t()[:, 1])
        q_m = torch.index_select(q_sum, 0, edge_index.t()[:, 0])
        for relation in range(self.num_relations):
            idx = edge_type == int(relation)
            if idx.sum() == 0:
                continue
            k_node_vec = k_m[idx].view(-1, self.n_heads, self.d_k)
            v_node_vec = v_m[idx].view(-1, self.n_heads, self.d_k)
            q_node_vec = q_m[idx].view(-1, self.n_heads, self.d_k)
            k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[relation]).transpose(1, 0)
            res_att[idx] = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[relation] / self.sqrt_dk
            res_msg[idx] = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[relation]).transpose(1, 0)
        del k_m
        del v_m
        del q_m
        gc.collect()
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        gc.collect()
        end = time.time()
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_maintype, type_count):
        start = time.time()
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        length = node_maintype.size(1)
        a_sum = 0
        a_ws = torch.matmul(self.a_att, self.a_basis.view(self.p.node_num_bases, -1))
        a_ws = a_ws.view(self.num_maintype, self.out_dim, self.out_dim)
        a_bs = torch.matmul(self.a_bias_att, self.a_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        a_m = self.drop(a_sum)
        if self.p.skip1:
            alpha1 = torch.index_select(self.skip1, 0, node_maintype).view(-1, 1)
        if self.p.skip2:
            alpha2 = torch.index_select(self.skip2, 0, node_maintype).view(-1, 1)
        if self.p.skip1:
            a_m = a_m * alpha1 + node_inp * (1 - alpha1)
        else:
            a_m = a_m + node_inp
        if self.use_norm:
            a_m = self.norms[0](a_m)
        mat = F.gelu(self.in_linear(a_m))
        for line in range(self.p.n_linear):
            mat = self.drop(F.gelu(self.hide_linears[line](mat)))
        mat = self.drop(self.out_linear(mat))
        if self.p.skip2:
            mat = mat * alpha2 + a_m * (1 - alpha2)
        else:
            mat = mat + a_m
        res = self.out_norm(mat)
        return res

    def rel_process(self, rel_inp, edge_index):
        rel_att = torch.zeros(self.num_relations, self.n_heads).to(rel_inp.device)
        rel_msg = torch.zeros(self.num_relations, self.n_heads, self.d_k).to(rel_inp.device)
        rk_m = self.rk_linear(rel_inp)
        rv_m = self.rv_linear(rel_inp)
        for idx in range(self.num_relations):
            rk_node_vec = rk_m[idx].view(-1, self.n_heads, self.d_k)
            rv_node_vec = rv_m[idx].view(-1, self.n_heads, self.d_k)
            rk_node_vec = torch.bmm(rk_node_vec.transpose(1, 0), self.rel_relation_att[idx]).transpose(1, 0)
            rel_att[idx] = rk_node_vec.sum(dim=-1) * self.rel_relation_pri[idx] / self.sqrt_dk
            rel_msg[idx] = torch.bmm(rv_node_vec.transpose(1, 0), self.rel_relation_msg[idx]).transpose(1, 0)
        att = torch.softmax(rel_att, dim=-1)
        rel = rel_msg * att.view(-1, self.n_heads, 1)
        rel = rel.view(-1, self.out_dim)
        del rel_att, rel_msg, rk_m, rv_m
        gc.collect()
        start = time.time()
        ra_m = F.gelu(rel)
        ra_m = self.ra_linear(ra_m)
        ra_m = self.drop(ra_m)
        if self.p.skip1:
            ra_m = ra_m * alpha1 + rel_inp * (1 - alpha1)
        else:
            ra_m = ra_m + rel_inp
        if self.use_norm:
            ra_m = self.rnorm(ra_m)
        rmat = F.gelu(self.rin_linear(ra_m))
        for line in range(self.p.n_linear):
            rmat = self.drop(F.gelu(self.rhide_linears[line](rmat)))
        rmat = self.drop(self.rout_linear(rmat))
        if self.p.skip2:
            rmat = rmat * alpha2 + ra_m * (1 - alpha2)
        else:
            rmat = rmat + ra_m
        rel = self.rout_norm(rmat)
        return rel

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class RelDenseMTHGCLConv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(RelDenseMTHGCLConv, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        self.num_type_bases = 5
        self.norms = nn.ModuleList()
        for t in range(1):
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.v_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.v_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.v_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.v_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.q_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.q_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.q_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.q_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.a_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim, out_dim)).cuda()
            self.a_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.a_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.a_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        self.drop = nn.Dropout(dropout)
        self.hide_linears = nn.ModuleList()
        for t in range(self.p.n_linear):
            self.hide_linears.append(nn.Linear(out_dim * self.p.dimrate_linear, out_dim * self.p.dimrate_linear))
        self.in_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.out_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        if self.p.skip1:
            self.skip1 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.skip2:
            self.skip2 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.re_num_bases > 0:
            self.at_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.at_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.ms_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.ms_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.pr_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads)).cuda()
            self.pr_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
        self.reset_parameters()
        self.relation_att = torch.matmul(self.at_att, self.at_basis.view(self.p.re_num_bases, -1))
        self.relation_att = self.relation_att.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
        self.relation_msg = torch.matmul(self.ms_att, self.ms_basis.view(self.p.re_num_bases, -1))
        self.relation_msg = self.relation_msg.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
        self.relation_pri = torch.matmul(self.pr_att, self.pr_basis)
        self.rk_linear = nn.Linear(self.in_dim, self.out_dim)
        self.ra_linear = nn.Linear(self.out_dim, self.out_dim)
        self.rnorm = nn.LayerNorm(out_dim)
        self.rin_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.rout_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.rout_norm = nn.LayerNorm(out_dim)

    def reset_parameters(self):
        node_size = self.p.node_num_bases * self.in_dim
        re_size = self.p.re_num_bases * self.n_heads
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.q_basis.data)
            xavier_normal_(self.q_att.data)
            xavier_normal_(self.q_bias.data)
            xavier_normal_(self.q_bias_att.data)
            xavier_normal_(self.v_basis.data)
            xavier_normal_(self.v_att.data)
            xavier_normal_(self.v_bias.data)
            xavier_normal_(self.v_bias_att.data)
            xavier_normal_(self.a_basis.data)
            xavier_normal_(self.a_att.data)
            xavier_normal_(self.a_bias.data)
            xavier_normal_(self.a_bias_att.data)
            xavier_normal_(self.at_basis.data)
            xavier_normal_(self.at_att.data)
            xavier_normal_(self.ms_basis.data)
            xavier_normal_(self.ms_att.data)
            xavier_normal_(self.pr_basis.data)
            xavier_normal_(self.pr_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            xavier_uniform_(self.q_basis.data)
            xavier_uniform_(self.q_att.data)
            get_uniform(node_size, self.q_bias)
            get_uniform(node_size, self.q_bias_att)
            xavier_uniform_(self.v_basis.data)
            xavier_uniform_(self.v_att.data)
            get_uniform(node_size, self.v_bias)
            get_uniform(node_size, self.v_bias_att)
            xavier_uniform_(self.a_basis.data)
            xavier_uniform_(self.a_att.data)
            get_uniform(node_size, self.a_bias)
            get_uniform(node_size, self.a_bias_att)
            xavier_uniform_(self.at_basis.data)
            xavier_uniform_(self.at_att.data)
            xavier_uniform_(self.ms_basis.data)
            xavier_uniform_(self.ms_att.data)
            xavier_uniform_(self.pr_basis.data)
            xavier_uniform_(self.pr_att.data)
        elif self.p.init_func == 'get_uniform':
            get_uniform(node_size, self.k_basis)
            get_uniform(node_size, self.k_att)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            get_uniform(node_size, self.q_basis)
            get_uniform(node_size, self.q_att)
            get_uniform(node_size, self.q_bias)
            get_uniform(node_size, self.q_bias_att)
            get_uniform(node_size, self.v_basis)
            get_uniform(node_size, self.v_att)
            get_uniform(node_size, self.v_bias)
            get_uniform(node_size, self.v_bias_att)
            get_uniform(node_size, self.a_basis)
            get_uniform(node_size, self.a_att)
            get_uniform(node_size, self.a_bias)
            get_uniform(node_size, self.a_bias_att)
            get_uniform(re_size, self.at_basis)
            get_uniform(re_size, self.at_att)
            get_uniform(re_size, self.ms_basis)
            get_uniform(re_size, self.ms_att)
            get_uniform(re_size, self.pr_basis)
            get_uniform(re_size, self.pr_att)
        elif self.p.init_func == 'get_normal':
            get_normal(self.k_basis)
            get_normal(self.k_att)
            get_normal(self.k_bias)
            get_normal(self.k_bias_att)
            get_normal(self.q_basis)
            get_normal(self.q_att)
            get_normal(self.q_bias)
            get_normal(self.q_bias_att)
            get_normal(self.v_basis)
            get_normal(self.v_att)
            get_normal(self.v_bias)
            get_normal(self.v_bias_att)
            get_normal(self.a_basis)
            get_normal(self.a_att)
            get_normal(self.a_bias)
            get_normal(self.a_bias_att)
            get_normal(self.at_basis)
            get_normal(self.at_att)
            get_normal(self.ms_basis)
            get_normal(self.ms_att)
            get_normal(self.pr_basis)
            get_normal(self.pr_att)
        else:
            raise Exception('input one of xavier_normal/xavier_uniform/get_uniform/get_normal')

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, rel_inp, adj):
        meta_xs = self.propagate(edge_index, node_inp=node_inp, node_maintype=node_maintype, type_count=type_count, edge_type=edge_type, rel_inp=rel_inp)
        meta_rel = self.rel_process(rel_inp, edge_index)
        return (meta_xs, meta_rel)

    def message(self, edge_index_i, edge_index, edge_type, node_inp, node_maintype, type_count):
        start = time.time()
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp.device)
        length = node_maintype.size(1)
        k_sum = 0
        v_sum = 0
        q_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        v_ws = torch.matmul(self.v_att, self.v_basis.view(self.p.node_num_bases, -1))
        v_ws = v_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        v_bs = torch.matmul(self.v_bias_att, self.v_bias)
        q_ws = torch.matmul(self.q_att, self.q_basis.view(self.p.node_num_bases, -1))
        q_ws = q_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        q_bs = torch.matmul(self.q_bias_att, self.q_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
                v_w = torch.index_select(v_ws, 0, node_maintype[:, i])
                v_b = torch.index_select(v_bs, 0, node_maintype[:, i])
                v_m = torch.bmm(node_inp.unsqueeze(1), v_w).squeeze(-2) + v_b
                v_sum = v_sum + v_m * type_count[:, i].view(-1, 1)
                q_w = torch.index_select(q_ws, 0, node_maintype[:, i])
                q_b = torch.index_select(q_bs, 0, node_maintype[:, i])
                q_m = torch.bmm(node_inp.unsqueeze(1), q_w).squeeze(-2) + q_b
                q_sum = q_sum + q_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
                v_w = torch.index_select(v_ws, 0, node_maintype[:, i])
                v_b = torch.index_select(v_bs, 0, node_maintype[:, i])
                v_m = torch.bmm(node_inp.unsqueeze(1), v_w).squeeze(-2) + v_b
                v_sum = v_sum + v_m * type_count[:, i].view(-1, 1)
                q_w = torch.index_select(q_ws, 0, node_maintype[:, i])
                q_b = torch.index_select(q_bs, 0, node_maintype[:, i])
                q_m = torch.bmm(node_inp.unsqueeze(1), q_w).squeeze(-2) + q_b
                q_sum = q_sum + q_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        k_m = torch.index_select(k_sum, 0, edge_index.t()[:, 1])
        v_m = torch.index_select(v_sum, 0, edge_index.t()[:, 1])
        q_m = torch.index_select(q_sum, 0, edge_index.t()[:, 0])
        for relation in range(self.num_relations):
            idx = edge_type == int(relation)
            if idx.sum() == 0:
                continue
            k_node_vec = k_m[idx].view(-1, self.n_heads, self.d_k)
            v_node_vec = v_m[idx].view(-1, self.n_heads, self.d_k)
            q_node_vec = q_m[idx].view(-1, self.n_heads, self.d_k)
            k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[relation]).transpose(1, 0)
            res_att[idx] = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[relation] / self.sqrt_dk
            res_msg[idx] = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[relation]).transpose(1, 0)
        del k_m
        del v_m
        del q_m
        gc.collect()
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        gc.collect()
        end = time.time()
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_maintype, type_count):
        start = time.time()
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        length = node_maintype.size(1)
        a_sum = 0
        a_ws = torch.matmul(self.a_att, self.a_basis.view(self.p.node_num_bases, -1))
        a_ws = a_ws.view(self.num_maintype, self.out_dim, self.out_dim)
        a_bs = torch.matmul(self.a_bias_att, self.a_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        a_m = self.drop(a_sum)
        if self.p.skip1:
            alpha1 = torch.index_select(self.skip1, 0, node_maintype).view(-1, 1)
        if self.p.skip2:
            alpha2 = torch.index_select(self.skip2, 0, node_maintype).view(-1, 1)
        if self.p.skip1:
            a_m = a_m * alpha1 + node_inp * (1 - alpha1)
        else:
            a_m = a_m + node_inp
        if self.use_norm:
            a_m = self.norms[0](a_m)
        mat = F.gelu(self.in_linear(a_m))
        for line in range(self.p.n_linear):
            mat = self.drop(F.gelu(self.hide_linears[line](mat)))
        mat = self.drop(self.out_linear(mat))
        if self.p.skip2:
            mat = mat * alpha2 + a_m * (1 - alpha2)
        else:
            mat = mat + a_m
        res = self.out_norm(mat)
        return res

    def rel_process(self, rel_inp, edge_index):
        rk_m = self.rk_linear(rel_inp)
        rel = []
        for idx in range(self.num_relations):
            rk_node_vec = rk_m[idx].view(-1, self.n_heads, self.d_k)
            rk_node_vec = torch.bmm(rk_node_vec.transpose(1, 0), self.relation_att[idx]).transpose(1, 0)
            rk_node_vec = rk_node_vec.view(-1)
            rel.append(rk_node_vec)
        rel = torch.stack(rel, dim=0)
        del rk_m
        gc.collect()
        start = time.time()
        ra_m = F.gelu(rel)
        ra_m = self.ra_linear(ra_m)
        ra_m = self.drop(ra_m)
        if self.p.skip1:
            ra_m = ra_m * alpha1 + rel_inp * (1 - alpha1)
        else:
            ra_m = ra_m + rel_inp
        if self.use_norm:
            ra_m = self.rnorm(ra_m)
        rmat = F.gelu(self.rin_linear(ra_m))
        for line in range(self.p.n_linear):
            rmat = self.drop(F.gelu(self.rhide_linears[line](rmat)))
        rmat = self.drop(self.rout_linear(rmat))
        if self.p.skip2:
            rmat = rmat * alpha2 + ra_m * (1 - alpha2)
        else:
            rmat = rmat + ra_m
        rel = self.rout_norm(rmat)
        return rel

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class RelDenseMTHGCLConv2(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(RelDenseMTHGCLConv2, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        self.num_type_bases = 5
        self.norms = nn.ModuleList()
        for t in range(1):
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
                self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.v_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.v_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.v_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
                self.v_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.q_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.q_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.q_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
                self.q_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.a_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim, out_dim)).cuda()
            self.a_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.a_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
                self.a_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        self.drop = nn.Dropout(dropout)
        self.hide_linears = nn.ModuleList()
        for t in range(self.p.n_linear):
            self.hide_linears.append(nn.Linear(out_dim * self.p.dimrate_linear, out_dim * self.p.dimrate_linear))
        self.in_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.out_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        if self.p.skip1:
            self.skip1 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.skip2:
            self.skip2 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.re_num_bases > 0:
            self.at_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.at_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.ms_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.ms_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.pr_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads)).cuda()
            self.pr_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
        self.reset_parameters()
        self.relation_att = torch.matmul(self.at_att, self.at_basis.view(self.p.re_num_bases, -1))
        self.relation_att = self.relation_att.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
        self.relation_msg = torch.matmul(self.ms_att, self.ms_basis.view(self.p.re_num_bases, -1))
        self.relation_msg = self.relation_msg.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
        self.relation_pri = torch.matmul(self.pr_att, self.pr_basis)
        self.rk_linear = nn.Linear(self.in_dim, self.out_dim)
        self.ra_linear = nn.Linear(self.out_dim, self.out_dim)
        self.rnorm = nn.LayerNorm(out_dim)
        self.rin_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.rout_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.rout_norm = nn.LayerNorm(out_dim)
        self.k_l = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.v_l = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.q_l = nn.Linear(self.out_dim, self.out_dim, bias=False)

    def reset_parameters(self):
        node_size = self.p.node_num_bases * self.in_dim
        re_size = self.p.re_num_bases * self.n_heads
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.k_bias.data)
                xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.q_basis.data)
            xavier_normal_(self.q_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.q_bias.data)
                xavier_normal_(self.q_bias_att.data)
            xavier_normal_(self.v_basis.data)
            xavier_normal_(self.v_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.v_bias.data)
                xavier_normal_(self.v_bias_att.data)
            xavier_normal_(self.a_basis.data)
            xavier_normal_(self.a_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.a_bias.data)
                xavier_normal_(self.a_bias_att.data)
            xavier_normal_(self.at_basis.data)
            xavier_normal_(self.at_att.data)
            xavier_normal_(self.ms_basis.data)
            xavier_normal_(self.ms_att.data)
            xavier_normal_(self.pr_basis.data)
            xavier_normal_(self.pr_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.k_bias)
                get_uniform(node_size, self.k_bias_att)
            xavier_uniform_(self.q_basis.data)
            xavier_uniform_(self.q_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.q_bias)
                get_uniform(node_size, self.q_bias_att)
            xavier_uniform_(self.v_basis.data)
            xavier_uniform_(self.v_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.v_bias)
                get_uniform(node_size, self.v_bias_att)
            xavier_uniform_(self.a_basis.data)
            xavier_uniform_(self.a_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.a_bias)
                get_uniform(node_size, self.a_bias_att)
            xavier_uniform_(self.at_basis.data)
            xavier_uniform_(self.at_att.data)
            xavier_uniform_(self.ms_basis.data)
            xavier_uniform_(self.ms_att.data)
            xavier_uniform_(self.pr_basis.data)
            xavier_uniform_(self.pr_att.data)
        elif self.p.init_func == 'get_uniform':
            get_uniform(node_size, self.k_basis)
            get_uniform(node_size, self.k_att)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            get_uniform(node_size, self.q_basis)
            get_uniform(node_size, self.q_att)
            get_uniform(node_size, self.q_bias)
            get_uniform(node_size, self.q_bias_att)
            get_uniform(node_size, self.v_basis)
            get_uniform(node_size, self.v_att)
            get_uniform(node_size, self.v_bias)
            get_uniform(node_size, self.v_bias_att)
            get_uniform(node_size, self.a_basis)
            get_uniform(node_size, self.a_att)
            get_uniform(node_size, self.a_bias)
            get_uniform(node_size, self.a_bias_att)
            get_uniform(re_size, self.at_basis)
            get_uniform(re_size, self.at_att)
            get_uniform(re_size, self.ms_basis)
            get_uniform(re_size, self.ms_att)
            get_uniform(re_size, self.pr_basis)
            get_uniform(re_size, self.pr_att)
        elif self.p.init_func == 'get_normal':
            get_normal(self.k_basis)
            get_normal(self.k_att)
            get_normal(self.k_bias)
            get_normal(self.k_bias_att)
            get_normal(self.q_basis)
            get_normal(self.q_att)
            get_normal(self.q_bias)
            get_normal(self.q_bias_att)
            get_normal(self.v_basis)
            get_normal(self.v_att)
            get_normal(self.v_bias)
            get_normal(self.v_bias_att)
            get_normal(self.a_basis)
            get_normal(self.a_att)
            get_normal(self.a_bias)
            get_normal(self.a_bias_att)
            get_normal(self.at_basis)
            get_normal(self.at_att)
            get_normal(self.ms_basis)
            get_normal(self.ms_att)
            get_normal(self.pr_basis)
            get_normal(self.pr_att)
        else:
            raise Exception('input one of xavier_normal/xavier_uniform/get_uniform/get_normal')

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, rel_inp, adj):
        meta_xs = self.propagate(edge_index, node_inp=node_inp, node_maintype=node_maintype, type_count=type_count, edge_type=edge_type, rel_inp=rel_inp)
        meta_rel = self.rel_process(rel_inp, edge_index)
        return (meta_xs, meta_rel)

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def message(self, edge_index_i, edge_index, edge_type, node_inp, node_maintype, type_count, rel_inp):
        start = time.time()
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp.device)
        length = node_maintype.size(1)
        k_sum = 0
        v_sum = 0
        q_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        if self.p.bias != 0:
            k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        v_ws = torch.matmul(self.v_att, self.v_basis.view(self.p.node_num_bases, -1))
        v_ws = v_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        if self.p.bias != 0:
            v_bs = torch.matmul(self.v_bias_att, self.v_bias)
        q_ws = torch.matmul(self.q_att, self.q_basis.view(self.p.node_num_bases, -1))
        q_ws = q_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        if self.p.bias != 0:
            q_bs = torch.matmul(self.q_bias_att, self.q_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                else:
                    k_b = 0
                k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
                v_w = torch.index_select(v_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    v_b = torch.index_select(v_bs, 0, node_maintype[:, i])
                else:
                    v_b = 0
                v_m = torch.bmm(node_inp.unsqueeze(1), v_w).squeeze(-2) + v_b
                v_sum = v_sum + v_m * type_count[:, i].view(-1, 1)
                q_w = torch.index_select(q_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    q_b = torch.index_select(q_bs, 0, node_maintype[:, i])
                else:
                    q_b = 0
                q_m = torch.bmm(node_inp.unsqueeze(1), q_w).squeeze(-2) + q_b
                q_sum = q_sum + q_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                else:
                    k_b = 0
                k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
                v_w = torch.index_select(v_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    v_b = torch.index_select(v_bs, 0, node_maintype[:, i])
                else:
                    v_b = 0
                v_m = torch.bmm(node_inp.unsqueeze(1), v_w).squeeze(-2) + v_b
                v_sum = v_sum + v_m * type_count[:, i].view(-1, 1)
                q_w = torch.index_select(q_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    q_b = torch.index_select(q_bs, 0, node_maintype[:, i])
                else:
                    q_b = 0
                q_m = torch.bmm(node_inp.unsqueeze(1), q_w).squeeze(-2) + q_b
                q_sum = q_sum + q_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        k_m = torch.index_select(k_sum, 0, edge_index.t()[:, 1])
        v_m = torch.index_select(v_sum, 0, edge_index.t()[:, 1])
        q_m = torch.index_select(q_sum, 0, edge_index.t()[:, 0])
        rel_emb = torch.index_select(rel_inp, 0, edge_type)
        k_m = self.rel_transform(k_m, rel_emb)
        v_m = self.rel_transform(v_m, rel_emb)
        q_m = self.rel_transform(q_m, rel_emb)
        k_m = self.k_l(k_m)
        v_m = self.v_l(v_m)
        q_m = self.q_l(q_m)
        for relation in range(self.num_relations):
            idx = edge_type == int(relation)
            if idx.sum() == 0:
                continue
            k_node_vec = k_m[idx].view(-1, self.n_heads, self.d_k)
            v_node_vec = v_m[idx].view(-1, self.n_heads, self.d_k)
            q_node_vec = q_m[idx].view(-1, self.n_heads, self.d_k)
            k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[relation]).transpose(1, 0)
            res_att[idx] = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[relation] / self.sqrt_dk
            res_msg[idx] = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[relation]).transpose(1, 0)
        del k_m
        del v_m
        del q_m
        gc.collect()
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        gc.collect()
        end = time.time()
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_maintype, type_count):
        start = time.time()
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        length = node_maintype.size(1)
        a_sum = 0
        a_ws = torch.matmul(self.a_att, self.a_basis.view(self.p.node_num_bases, -1))
        a_ws = a_ws.view(self.num_maintype, self.out_dim, self.out_dim)
        if self.p.bias != 0:
            a_bs = torch.matmul(self.a_bias_att, self.a_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                else:
                    a_b = 0
                a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                else:
                    a_b = 0
                a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        a_m = self.drop(a_sum)
        if self.p.skip1:
            alpha1 = torch.index_select(self.skip1, 0, node_maintype).view(-1, 1)
        if self.p.skip2:
            alpha2 = torch.index_select(self.skip2, 0, node_maintype).view(-1, 1)
        if self.p.skip1:
            a_m = a_m * alpha1 + node_inp * (1 - alpha1)
        else:
            a_m = a_m + node_inp
        if self.use_norm:
            a_m = self.norms[0](a_m)
        mat = F.gelu(self.in_linear(a_m))
        for line in range(self.p.n_linear):
            mat = self.drop(F.gelu(self.hide_linears[line](mat)))
        mat = self.drop(self.out_linear(mat))
        if self.p.skip2:
            mat = mat * alpha2 + a_m * (1 - alpha2)
        else:
            mat = mat + a_m
        res = self.out_norm(mat)
        return res

    def rel_process(self, rel_inp, edge_index):
        rk_m = self.rk_linear(rel_inp)
        rel = []
        for idx in range(self.num_relations):
            rk_node_vec = rk_m[idx].view(-1, self.n_heads, self.d_k)
            rk_node_vec = torch.bmm(rk_node_vec.transpose(1, 0), self.relation_att[idx]).transpose(1, 0)
            rk_node_vec = rk_node_vec.view(-1)
            rel.append(rk_node_vec)
        rel = torch.stack(rel, dim=0)
        del rk_m
        gc.collect()
        start = time.time()
        ra_m = rel
        ra_m = self.ra_linear(ra_m)
        ra_m = self.drop(ra_m)
        if self.p.skip1:
            ra_m = ra_m * alpha1 + rel_inp * (1 - alpha1)
        else:
            ra_m = ra_m + rel_inp
        rmat = self.rin_linear(ra_m)
        for line in range(self.p.n_linear):
            rmat = self.drop(self.rhide_linears[line](rmat))
        rmat = self.drop(self.rout_linear(rmat))
        if self.p.skip2:
            rmat = rmat * alpha2 + ra_m * (1 - alpha2)
        else:
            rmat = rmat + ra_m
        rel = rmat
        return rel

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class LGTNConv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(LGTNConv, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        self.num_type_bases = 5
        self.norms = nn.ModuleList()
        for t in range(1):
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
                self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.a_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim, out_dim)).cuda()
            self.a_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.a_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
                self.a_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        self.drop = nn.Dropout(dropout)
        self.hide_linears = nn.ModuleList()
        for t in range(self.p.n_linear):
            self.hide_linears.append(nn.Linear(out_dim * self.p.dimrate_linear, out_dim * self.p.dimrate_linear))
        self.in_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.out_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        if self.p.skip1:
            self.skip1 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.skip2:
            self.skip2 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.re_num_bases > 0:
            self.at_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.at_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.ms_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.ms_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.pr_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads)).cuda()
            self.pr_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
        self.reset_parameters()
        self.relation_att = torch.matmul(self.at_att, self.at_basis.view(self.p.re_num_bases, -1))
        self.relation_att = self.relation_att.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
        self.relation_msg = torch.matmul(self.ms_att, self.ms_basis.view(self.p.re_num_bases, -1))
        self.relation_msg = self.relation_msg.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
        self.relation_pri = torch.matmul(self.pr_att, self.pr_basis)
        self.rin_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.rout_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.k_l = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.v_l = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.q_l = nn.Linear(self.out_dim, self.out_dim, bias=False)

    def reset_parameters(self):
        node_size = self.p.node_num_bases * self.in_dim
        re_size = self.p.re_num_bases * self.n_heads
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.k_bias.data)
                xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.a_basis.data)
            xavier_normal_(self.a_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.a_bias.data)
                xavier_normal_(self.a_bias_att.data)
            xavier_normal_(self.at_basis.data)
            xavier_normal_(self.at_att.data)
            xavier_normal_(self.ms_basis.data)
            xavier_normal_(self.ms_att.data)
            xavier_normal_(self.pr_basis.data)
            xavier_normal_(self.pr_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.k_bias)
                get_uniform(node_size, self.k_bias_att)
            xavier_uniform_(self.a_basis.data)
            xavier_uniform_(self.a_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.a_bias)
                get_uniform(node_size, self.a_bias_att)
            xavier_uniform_(self.at_basis.data)
            xavier_uniform_(self.at_att.data)
            xavier_uniform_(self.ms_basis.data)
            xavier_uniform_(self.ms_att.data)
            xavier_uniform_(self.pr_basis.data)
            xavier_uniform_(self.pr_att.data)
        elif self.p.init_func == 'get_uniform':
            get_uniform(node_size, self.k_basis)
            get_uniform(node_size, self.k_att)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            get_uniform(node_size, self.q_basis)
            get_uniform(node_size, self.q_att)
            get_uniform(node_size, self.q_bias)
            get_uniform(node_size, self.q_bias_att)
            get_uniform(node_size, self.v_basis)
            get_uniform(node_size, self.v_att)
            get_uniform(node_size, self.v_bias)
            get_uniform(node_size, self.v_bias_att)
            get_uniform(node_size, self.a_basis)
            get_uniform(node_size, self.a_att)
            get_uniform(node_size, self.a_bias)
            get_uniform(node_size, self.a_bias_att)
            get_uniform(re_size, self.at_basis)
            get_uniform(re_size, self.at_att)
            get_uniform(re_size, self.ms_basis)
            get_uniform(re_size, self.ms_att)
            get_uniform(re_size, self.pr_basis)
            get_uniform(re_size, self.pr_att)
        elif self.p.init_func == 'get_normal':
            get_normal(self.k_basis)
            get_normal(self.k_att)
            get_normal(self.k_bias)
            get_normal(self.k_bias_att)
            get_normal(self.q_basis)
            get_normal(self.q_att)
            get_normal(self.q_bias)
            get_normal(self.q_bias_att)
            get_normal(self.v_basis)
            get_normal(self.v_att)
            get_normal(self.v_bias)
            get_normal(self.v_bias_att)
            get_normal(self.a_basis)
            get_normal(self.a_att)
            get_normal(self.a_bias)
            get_normal(self.a_bias_att)
            get_normal(self.at_basis)
            get_normal(self.at_att)
            get_normal(self.ms_basis)
            get_normal(self.ms_att)
            get_normal(self.pr_basis)
            get_normal(self.pr_att)
        else:
            raise Exception('input one of xavier_normal/xavier_uniform/get_uniform/get_normal')

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, rel_inp, adj):
        meta_xs = self.propagate(edge_index, node_inp=node_inp, node_maintype=node_maintype, type_count=type_count, edge_type=edge_type, rel_inp=rel_inp)
        meta_rel = self.rel_process(rel_inp, edge_index)
        return (meta_xs, meta_rel)

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def message(self, edge_index_i, edge_index, edge_type, node_inp, node_maintype, type_count, rel_inp):
        start = time.time()
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp.device)
        length = node_maintype.size(1)
        k_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        if self.p.bias != 0:
            k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                else:
                    k_b = 0
                k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                else:
                    k_b = 0
                k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        k_m = torch.index_select(k_sum, 0, edge_index.t()[:, 1])
        v_m = k_m
        q_m = torch.index_select(k_sum, 0, edge_index.t()[:, 0])
        rel_emb = torch.index_select(rel_inp, 0, edge_type)
        k_m = self.rel_transform(k_m, rel_emb)
        v_m = self.rel_transform(v_m, rel_emb)
        q_m = self.rel_transform(q_m, rel_emb)
        k_m = self.k_l(k_m)
        v_m = self.v_l(v_m)
        q_m = self.q_l(q_m)
        for relation in range(self.num_relations):
            idx = edge_type == int(relation)
            if idx.sum() == 0:
                continue
            k_node_vec = k_m[idx].view(-1, self.n_heads, self.d_k)
            v_node_vec = v_m[idx].view(-1, self.n_heads, self.d_k)
            q_node_vec = q_m[idx].view(-1, self.n_heads, self.d_k)
            k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[relation]).transpose(1, 0)
            res_att[idx] = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[relation] / self.sqrt_dk
            res_msg[idx] = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[relation]).transpose(1, 0)
        del k_m
        del v_m
        del q_m
        gc.collect()
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        gc.collect()
        end = time.time()
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_maintype, type_count):
        start = time.time()
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        length = node_maintype.size(1)
        a_sum = 0
        a_ws = torch.matmul(self.a_att, self.a_basis.view(self.p.node_num_bases, -1))
        a_ws = a_ws.view(self.num_maintype, self.out_dim, self.out_dim)
        if self.p.bias != 0:
            a_bs = torch.matmul(self.a_bias_att, self.a_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                else:
                    a_b = 0
                a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                else:
                    a_b = 0
                a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        a_m = self.drop(a_sum)
        if self.p.skip1:
            alpha1 = torch.index_select(self.skip1, 0, node_maintype).view(-1, 1)
        if self.p.skip2:
            alpha2 = torch.index_select(self.skip2, 0, node_maintype).view(-1, 1)
        if self.p.skip1:
            a_m = a_m * alpha1 + node_inp * (1 - alpha1)
        else:
            a_m = a_m + node_inp
        if self.use_norm:
            a_m = self.norms[0](a_m)
        mat = F.gelu(self.in_linear(a_m))
        for line in range(self.p.n_linear):
            mat = self.drop(F.gelu(self.hide_linears[line](mat)))
        mat = self.drop(self.out_linear(mat))
        if self.p.skip2:
            mat = mat * alpha2 + a_m * (1 - alpha2)
        else:
            mat = mat + a_m
        res = self.out_norm(mat)
        return res

    def rel_process(self, rel_inp, edge_index):
        rel = []
        for idx in range(self.num_relations):
            rk_node_vec = rel_inp[idx].view(-1, self.n_heads, self.d_k)
            rk_node_vec = torch.bmm(rk_node_vec.transpose(1, 0), self.relation_att[idx]).transpose(1, 0)
            rk_node_vec = rk_node_vec.view(-1)
            rel.append(rk_node_vec)
        rel = torch.stack(rel, dim=0)
        gc.collect()
        start = time.time()
        ra_m = F.gelu(rel)
        ra_m = self.drop(ra_m)
        if self.p.skip1:
            ra_m = ra_m * alpha1 + rel_inp * (1 - alpha1)
        else:
            ra_m = ra_m + rel_inp
        rmat = F.gelu(self.rin_linear(ra_m))
        for line in range(self.p.n_linear):
            rmat = self.drop(F.gelu(self.rhide_linears[line](rmat)))
        rmat = self.drop(self.rout_linear(rmat))
        if self.p.skip2:
            rmat = rmat * alpha2 + ra_m * (1 - alpha2)
        else:
            rmat = rmat + ra_m
        rel = rmat
        return rel

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class LGTNv2Conv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(LGTNv2Conv, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        self.num_type_bases = 5
        self.norms = nn.ModuleList()
        for t in range(1):
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
                self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.a_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim, out_dim)).cuda()
            self.a_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.a_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
                self.a_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        self.drop = nn.Dropout(dropout)
        self.hide_linears = nn.ModuleList()
        for t in range(self.p.n_linear):
            self.hide_linears.append(nn.Linear(out_dim * self.p.dimrate_linear, out_dim * self.p.dimrate_linear))
        self.in_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.out_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        if self.p.skip1:
            self.skip1 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.skip2:
            self.skip2 = nn.Parameter(torch.ones(self.num_maintype))
        self.relation_att = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_k, self.d_k)).cuda()
        self.relation_msg = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_k, self.d_k)).cuda()
        self.relation_pri = nn.Parameter(torch.Tensor(1, self.n_heads)).cuda()
        self.reset_parameters()
        self.rin_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.rout_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.k_l = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.v_l = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.q_l = nn.Linear(self.out_dim, self.out_dim, bias=False)

    def reset_parameters(self):
        node_size = self.p.node_num_bases * self.in_dim
        re_size = self.p.re_num_bases * self.n_heads
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.k_bias.data)
                xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.a_basis.data)
            xavier_normal_(self.a_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.a_bias.data)
                xavier_normal_(self.a_bias_att.data)
            xavier_normal_(self.at_basis.data)
            xavier_normal_(self.at_att.data)
            xavier_normal_(self.ms_basis.data)
            xavier_normal_(self.ms_att.data)
            xavier_normal_(self.pr_basis.data)
            xavier_normal_(self.pr_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.k_bias)
                get_uniform(node_size, self.k_bias_att)
            xavier_uniform_(self.a_basis.data)
            xavier_uniform_(self.a_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.a_bias)
                get_uniform(node_size, self.a_bias_att)
            xavier_uniform_(self.relation_att.data)
            xavier_uniform_(self.relation_msg.data)
            xavier_uniform_(self.relation_pri.data)
        elif self.p.init_func == 'get_uniform':
            get_uniform(node_size, self.k_basis)
            get_uniform(node_size, self.k_att)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            get_uniform(node_size, self.q_basis)
            get_uniform(node_size, self.q_att)
            get_uniform(node_size, self.q_bias)
            get_uniform(node_size, self.q_bias_att)
            get_uniform(node_size, self.v_basis)
            get_uniform(node_size, self.v_att)
            get_uniform(node_size, self.v_bias)
            get_uniform(node_size, self.v_bias_att)
            get_uniform(node_size, self.a_basis)
            get_uniform(node_size, self.a_att)
            get_uniform(node_size, self.a_bias)
            get_uniform(node_size, self.a_bias_att)
            get_uniform(re_size, self.at_basis)
            get_uniform(re_size, self.at_att)
            get_uniform(re_size, self.ms_basis)
            get_uniform(re_size, self.ms_att)
            get_uniform(re_size, self.pr_basis)
            get_uniform(re_size, self.pr_att)
        elif self.p.init_func == 'get_normal':
            get_normal(self.k_basis)
            get_normal(self.k_att)
            get_normal(self.k_bias)
            get_normal(self.k_bias_att)
            get_normal(self.q_basis)
            get_normal(self.q_att)
            get_normal(self.q_bias)
            get_normal(self.q_bias_att)
            get_normal(self.v_basis)
            get_normal(self.v_att)
            get_normal(self.v_bias)
            get_normal(self.v_bias_att)
            get_normal(self.a_basis)
            get_normal(self.a_att)
            get_normal(self.a_bias)
            get_normal(self.a_bias_att)
            get_normal(self.at_basis)
            get_normal(self.at_att)
            get_normal(self.ms_basis)
            get_normal(self.ms_att)
            get_normal(self.pr_basis)
            get_normal(self.pr_att)
        else:
            raise Exception('input one of xavier_normal/xavier_uniform/get_uniform/get_normal')

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, rel_inp, adj):
        meta_xs = self.propagate(edge_index, node_inp=node_inp, node_maintype=node_maintype, type_count=type_count, edge_type=edge_type, rel_inp=rel_inp)
        meta_rel = self.rel_process(rel_inp, edge_index)
        return (meta_xs, meta_rel)

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def message(self, edge_index_i, edge_index, edge_type, node_inp, node_maintype, type_count, rel_inp):
        start = time.time()
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp.device)
        length = node_maintype.size(1)
        k_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        if self.p.bias != 0:
            k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                else:
                    k_b = 0
                k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                else:
                    k_b = 0
                k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        k_m = torch.index_select(k_sum, 0, edge_index.t()[:, 1])
        v_m = torch.index_select(k_sum, 0, edge_index.t()[:, 1])
        q_m = torch.index_select(k_sum, 0, edge_index.t()[:, 0])
        rel_emb = torch.index_select(rel_inp, 0, edge_type)
        k_m = self.rel_transform(k_m, rel_emb)
        v_m = self.rel_transform(v_m, rel_emb)
        q_m = self.rel_transform(q_m, rel_emb)
        k_m = self.k_l(k_m)
        v_m = self.v_l(v_m)
        q_m = self.q_l(q_m)
        k_node_vec = k_m.view(-1, self.n_heads, self.d_k)
        v_node_vec = v_m.view(-1, self.n_heads, self.d_k)
        q_node_vec = q_m.view(-1, self.n_heads, self.d_k)
        k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[0]).transpose(1, 0)
        res_att = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[0] / self.sqrt_dk
        res_msg = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[0]).transpose(1, 0)
        del k_m
        del v_m
        del q_m
        gc.collect()
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        gc.collect()
        end = time.time()
        return res.reshape(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_maintype, type_count):
        start = time.time()
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        length = node_maintype.size(1)
        a_sum = 0
        a_ws = torch.matmul(self.a_att, self.a_basis.view(self.p.node_num_bases, -1))
        a_ws = a_ws.view(self.num_maintype, self.out_dim, self.out_dim)
        if self.p.bias != 0:
            a_bs = torch.matmul(self.a_bias_att, self.a_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                else:
                    a_b = 0
                a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                else:
                    a_b = 0
                a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        a_m = self.drop(a_sum)
        if self.p.skip1:
            alpha1 = torch.index_select(self.skip1, 0, node_maintype).view(-1, 1)
        if self.p.skip2:
            alpha2 = torch.index_select(self.skip2, 0, node_maintype).view(-1, 1)
        if self.p.skip1:
            a_m = a_m * alpha1 + node_inp * (1 - alpha1)
        else:
            a_m = a_m + node_inp
        if self.use_norm:
            a_m = self.norms[0](a_m)
        mat = F.gelu(self.in_linear(a_m))
        for line in range(self.p.n_linear):
            mat = self.drop(F.gelu(self.hide_linears[line](mat)))
        mat = self.drop(self.out_linear(mat))
        if self.p.skip2:
            mat = mat * alpha2 + a_m * (1 - alpha2)
        else:
            mat = mat + a_m
        res = self.out_norm(mat)
        return res

    def rel_process(self, rel_inp, edge_index):
        rk_node_vec = rel_inp.view(-1, self.n_heads, self.d_k)
        rk_node_vec = torch.bmm(rk_node_vec.transpose(1, 0), self.relation_att[0]).transpose(1, 0)
        rel = rk_node_vec.contiguous().view(self.num_relations, -1)
        start = time.time()
        ra_m = F.gelu(rel)
        ra_m = self.drop(ra_m)
        if self.p.skip1:
            ra_m = ra_m * alpha1 + rel_inp * (1 - alpha1)
        else:
            ra_m = ra_m + rel_inp
        rmat = F.gelu(self.rin_linear(ra_m))
        for line in range(self.p.n_linear):
            rmat = self.drop(F.gelu(self.rhide_linears[line](rmat)))
        rmat = self.drop(self.rout_linear(rmat))
        if self.p.skip2:
            rmat = rmat * alpha2 + ra_m * (1 - alpha2)
        else:
            rmat = rmat + ra_m
        rel = rmat
        return rel

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class GTNConv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(GTNConv, self).__init__(node_dim=0, aggr='add', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        self.num_type_bases = 5
        self.n_norms = nn.ModuleList()
        self.r_norms = nn.ModuleList()
        if self.p.mode_norm == 'BN':
            self.update_norm = nn.BatchNorm1d(out_dim)
            self.all_out_norm = nn.BatchNorm1d(out_dim)
            for t in range(3):
                if use_norm:
                    self.n_norms.append(nn.BatchNorm1d(out_dim))
            for t in range(1):
                if use_norm:
                    self.r_norms.append(nn.BatchNorm1d(out_dim))
        elif self.p.mode_norm == 'LN':
            self.update_norm = nn.LayerNorm(out_dim)
            self.all_out_norm = nn.LayerNorm(out_dim)
            for t in range(3):
                if use_norm:
                    self.n_norms.append(nn.LayerNorm(out_dim))
            for t in range(1):
                if use_norm:
                    self.r_norms.append(nn.LayerNorm(out_dim))
        if self.p.mode_activation == 'tanh':
            self.act_f = F.tanh
        elif self.p.mode_activation == 'gelu':
            self.act_f = F.gelu
        self.k_linear_h = nn.Linear(in_dim, out_dim)
        self.q_linear_h = nn.Linear(in_dim, out_dim)
        self.v_linear_h = nn.Linear(in_dim, out_dim)
        self.a_linear = nn.Linear(in_dim, out_dim)
        self.k_linear_t = nn.Linear(in_dim, out_dim)
        self.q_linear_t = nn.Linear(in_dim, out_dim)
        self.v_linear_t = nn.Linear(in_dim, out_dim)
        self.loop = nn.Linear(in_dim, out_dim)
        self.r_linear = nn.Linear(in_dim, out_dim)
        self.relation_att = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_k, self.d_k)).cuda()
        self.relation_msg = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_k, self.d_k)).cuda()
        self.relation_pri = nn.Parameter(torch.Tensor(1, self.n_heads)).cuda()
        xavier_uniform_(self.relation_att.data)
        xavier_uniform_(self.relation_msg.data)
        xavier_uniform_(self.relation_pri.data)
        self.drop = nn.Dropout(dropout)
        if self.p.mode_skip == 'dense':
            self.mid_linear = nn.Linear(out_dim, out_dim * 2)
            self.out_linear = nn.Linear(out_dim * 2, out_dim)
        self.skip = nn.Parameter(torch.ones(1))

    def reset_parameters(self):
        node_size = self.out_dim
        re_size = self.n_heads
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_linear_h.data)
            xavier_normal_(self.q_linear_h.data)
            xavier_normal_(self.v_linear_h.data)
            xavier_normal_(self.k_linear_t.data)
            xavier_normal_(self.q_linear_t.data)
            xavier_normal_(self.v_linear_t.data)
            xavier_normal_(self.a_linear.data)
            xavier_normal_(self.r_linear.data)
            if self.p.bias != 0:
                xavier_normal_(self.k_bias_h.data)
                xavier_normal_(self.q_bias_h.data)
                xavier_normal_(self.v_bias_h.data)
                xavier_normal_(self.k_bias_t.data)
                xavier_normal_(self.q_bias_t.data)
                xavier_normal_(self.v_bias_t.data)
                xavier_normal_(self.a_bias.data)
                xavier_normal_(self.r_bias.data)
            xavier_normal_(self.relation_att.data)
            xavier_normal_(self.relation_msg.data)
            xavier_normal_(self.relation_pri.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_linear_h.data)
            xavier_uniform_(self.q_linear_h.data)
            xavier_uniform_(self.v_linear_h.data)
            xavier_uniform_(self.k_linear_t.data)
            xavier_uniform_(self.q_linear_t.data)
            xavier_uniform_(self.v_linear_t.data)
            xavier_uniform_(self.a_linear.data)
            xavier_uniform_(self.r_linear.data)
            if self.p.bias != 0:
                xavier_uniform_(self.k_bias_h.data)
                xavier_uniform_(self.q_bias_h.data)
                xavier_uniform_(self.v_bias_h.data)
                xavier_uniform_(self.k_bias_t.data)
                xavier_uniform_(self.q_bias_t.data)
                xavier_uniform_(self.v_bias_t.data)
                xavier_uniform_(self.a_bias.data)
                xavier_uniform_(self.r_bias.data)
            xavier_uniform_(self.relation_att.data)
            xavier_uniform_(self.relation_msg.data)
            xavier_uniform_(self.relation_pri.data)
        elif self.p.init_func == 'get_uniform':
            get_uniform(node_size, self.k_linear_h)
            get_uniform(node_size, self.q_linear_h)
            get_uniform(node_size, self.v_linear_h)
            get_uniform(node_size, self.k_linear_t)
            get_uniform(node_size, self.q_linear_t)
            get_uniform(node_size, self.v_linear_t)
            get_uniform(node_size, self.a_linear)
            get_uniform(node_size, self.r_linear)
            if self.p.bias != 0:
                get_uniform(node_size, sself.k_bias_h)
                get_uniform(node_size, sself.q_bias_h)
                get_uniform(node_size, sself.v_bias_h)
                get_uniform(node_size, sself.k_bias_t)
                get_uniform(node_size, sself.q_bias_t)
                get_uniform(node_size, sself.v_bias_t)
                get_uniform(node_size, sself.a_bias)
                get_uniform(node_size, sself.r_bias)
            get_uniform(re_size, self.relation_att)
            get_uniform(re_size, self.relation_msg)
            get_uniform(re_size, self.relation_pri)
        elif self.p.init_func == 'get_normal':
            get_normal(self.k_linear_h)
            get_normal(self.q_linear_h)
            get_normal(self.v_linear_h)
            get_normal(self.k_linear_t)
            get_normal(self.q_linear_t)
            get_normal(self.v_linear_t)
            get_normal(self.a_linear)
            get_normal(self.r_linear)
            if self.p.bias != 0:
                get_normal(self.k_bias_h)
                get_normal(self.q_bias_h)
                get_normal(self.v_bias_h)
                get_normal(self.k_bias_t)
                get_normal(self.q_bias_t)
                get_normal(self.v_bias_t)
                get_normal(self.a_bias)
                get_normal(self.r_bias)
            get_normal(self.relation_att)
            get_normal(self.relation_msg)
            get_normal(self.relation_pri)
        else:
            raise Exception('input one of xavier_normal/xavier_uniform/get_uniform/get_normal')

    def forward(self, node_embed, node_type, type_count, edge_index, edge_type, rel_embed, adj):
        loop_index = edge_index.size(1) - node_embed.size(0)
        num_edges = loop_index // 2
        num_ent = node_embed.size(0)
        self.in_index, self.out_index = (edge_index[:, :num_edges], edge_index[:, num_edges:loop_index])
        self.in_type, self.out_type = (edge_type[:num_edges], edge_type[num_edges:loop_index])
        self.loop_index = edge_index[:, loop_index:]
        self.loop_type = edge_type[loop_index:]
        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)
        in_res = self.propagate(edge_index=self.in_index, node_embed=node_embed, edge_type=self.in_type, rel_embed=rel_embed, node_type=node_type, type_count=type_count, edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate(edge_index=self.loop_index, node_embed=node_embed, edge_type=self.loop_type, rel_embed=rel_embed, node_type=node_type, type_count=type_count, edge_norm=None, mode='loop')
        out_res = self.propagate(edge_index=self.out_index, node_embed=node_embed, edge_type=self.out_type, rel_embed=rel_embed, node_type=node_type, type_count=type_count, edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)
        return (out, self.drop(self.act_f(self.r_linear(rel_embed))))

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        return norm

    def message(self, edge_index, edge_type, edge_index_i, edge_index_j, edge_norm, mode, rel_embed, node_embed, node_embed_j, node_type, node_type_j, type_count):
        data_size = edge_index_j.size(0)
        r_out = rel_embed
        rel_emb = torch.index_select(r_out, 0, edge_type)
        if mode == 'in':
            x = node_embed
            k_out = self.k_linear_h(x)
            q_out = self.q_linear_h(x)
            v_out = self.v_linear_h(x)
        elif mode == 'out':
            x = node_embed
            k_out = self.k_linear_t(x)
            q_out = self.q_linear_t(x)
            v_out = self.v_linear_t(x)
        elif mode == 'loop':
            x = node_embed
            k_out = q_out = v_out = self.loop(x)
        else:
            exit('mode error')
        k_m = torch.index_select(k_out, 0, edge_index.t()[:, 1])
        q_m = torch.index_select(q_out, 0, edge_index.t()[:, 0])
        v_m = torch.index_select(v_out, 0, edge_index.t()[:, 1])
        k_m = self.rel_transform(k_m, rel_emb)
        v_m = self.rel_transform(v_m, rel_emb)
        k_node_vec = k_m.view(-1, self.n_heads, self.d_k)
        q_node_vec = q_m.view(-1, self.n_heads, self.d_k)
        v_node_vec = v_m.view(-1, self.n_heads, self.d_k)
        k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[0]).transpose(1, 0)
        res_att = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[0] / self.sqrt_dk
        res_msg = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[0]).transpose(1, 0)
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        return res.reshape(-1, self.out_dim)

    def update(self, aggr_out, node_embed, node_type, type_count, mode):
        x = aggr_out
        a_m = self.a_linear(x)
        if mode == 'in':
            if self.use_norm:
                a_m = self.n_norms[0](a_m)
        elif mode == 'loop':
            if self.use_norm:
                a_m = self.n_norms[1](a_m)
        elif mode == 'out':
            if self.use_norm:
                a_m = self.n_norms[2](a_m)
        res = self.act_f(a_m)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class LGTNv3Conv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(LGTNv3Conv, self).__init__(node_dim=0, aggr='add', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        self.num_type_bases = 5
        self.n_norms = nn.ModuleList()
        self.r_norms = nn.ModuleList()
        if self.p.mode_norm == 'BN':
            self.update_norm = nn.BatchNorm1d(out_dim)
            self.all_out_norm = nn.BatchNorm1d(out_dim)
            for t in range(3):
                if use_norm:
                    self.n_norms.append(nn.BatchNorm1d(out_dim))
            for t in range(1):
                if use_norm:
                    self.r_norms.append(nn.BatchNorm1d(out_dim))
        elif self.p.mode_norm == 'LN':
            self.update_norm = nn.LayerNorm(out_dim)
            self.all_out_norm = nn.LayerNorm(out_dim)
            for t in range(3):
                if use_norm:
                    self.n_norms.append(nn.LayerNorm(out_dim))
            for t in range(1):
                if use_norm:
                    self.r_norms.append(nn.LayerNorm(out_dim))
        if self.p.mode_activation == 'tanh':
            self.act_f = F.tanh
        elif self.p.mode_activation == 'gelu':
            self.act_f = F.gelu
        self.k_linear_h = nn.Parameter(torch.Tensor(self.num_maintype, out_dim)).cuda()
        self.q_linear_h = nn.Parameter(torch.Tensor(self.num_maintype, out_dim)).cuda()
        self.v_linear_h = nn.Parameter(torch.Tensor(self.num_maintype, out_dim)).cuda()
        self.a_linear = nn.Parameter(torch.Tensor(self.num_maintype, out_dim)).cuda()
        if self.p.bias != 0:
            self.k_bias_h = nn.Parameter(torch.Tensor(out_dim)).cuda()
            self.q_bias_h = nn.Parameter(torch.Tensor(out_dim)).cuda()
            self.v_bias_h = nn.Parameter(torch.Tensor(out_dim)).cuda()
            self.a_bias = nn.Parameter(torch.Tensor(out_dim)).cuda()
        else:
            self.k_bias_h = 0
            self.q_bias_h = 0
            self.v_bias_h = 0
            self.a_bias = 0
        self.k_linear_t = nn.Parameter(torch.Tensor(self.num_maintype, out_dim)).cuda()
        self.q_linear_t = nn.Parameter(torch.Tensor(self.num_maintype, out_dim)).cuda()
        self.v_linear_t = nn.Parameter(torch.Tensor(self.num_maintype, out_dim)).cuda()
        if self.p.bias != 0:
            self.k_bias_t = nn.Parameter(torch.Tensor(out_dim)).cuda()
            self.q_bias_t = nn.Parameter(torch.Tensor(out_dim)).cuda()
            self.v_bias_t = nn.Parameter(torch.Tensor(out_dim)).cuda()
        else:
            self.k_bias_t = 0
            self.q_bias_t = 0
            self.v_bias_t = 0
        self.loop = nn.Linear(in_dim, out_dim)
        self.r_linear = nn.Parameter(torch.Tensor(self.num_relations, out_dim)).cuda()
        if self.p.bias != 0:
            self.r_bias = nn.Parameter(torch.Tensor(out_dim)).cuda()
        else:
            self.r_bias = 0
        self.relation_att = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_k, self.d_k)).cuda()
        self.relation_msg = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_k, self.d_k)).cuda()
        self.relation_pri = nn.Parameter(torch.Tensor(1, self.n_heads)).cuda()
        self.drop = nn.Dropout(dropout)
        if self.p.mode_skip == 'dense':
            self.mid_linear = nn.Linear(out_dim, out_dim * 2)
            self.out_linear = nn.Linear(out_dim * 2, out_dim)
        self.skip = nn.Parameter(torch.ones(1))
        self.reset_parameters()

    def reset_parameters(self):
        node_size = self.out_dim
        re_size = self.n_heads
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_linear_h.data)
            xavier_normal_(self.q_linear_h.data)
            xavier_normal_(self.v_linear_h.data)
            xavier_normal_(self.k_linear_t.data)
            xavier_normal_(self.q_linear_t.data)
            xavier_normal_(self.v_linear_t.data)
            xavier_normal_(self.a_linear.data)
            xavier_normal_(self.r_linear.data)
            if self.p.bias != 0:
                xavier_normal_(self.k_bias_h.data)
                xavier_normal_(self.q_bias_h.data)
                xavier_normal_(self.v_bias_h.data)
                xavier_normal_(self.k_bias_t.data)
                xavier_normal_(self.q_bias_t.data)
                xavier_normal_(self.v_bias_t.data)
                xavier_normal_(self.a_bias.data)
                xavier_normal_(self.r_bias.data)
            xavier_normal_(self.relation_att.data)
            xavier_normal_(self.relation_msg.data)
            xavier_normal_(self.relation_pri.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_linear_h.data)
            xavier_uniform_(self.q_linear_h.data)
            xavier_uniform_(self.v_linear_h.data)
            xavier_uniform_(self.k_linear_t.data)
            xavier_uniform_(self.q_linear_t.data)
            xavier_uniform_(self.v_linear_t.data)
            xavier_uniform_(self.a_linear.data)
            xavier_uniform_(self.r_linear.data)
            if self.p.bias != 0:
                xavier_uniform_(self.k_bias_h.data)
                xavier_uniform_(self.q_bias_h.data)
                xavier_uniform_(self.v_bias_h.data)
                xavier_uniform_(self.k_bias_t.data)
                xavier_uniform_(self.q_bias_t.data)
                xavier_uniform_(self.v_bias_t.data)
                xavier_uniform_(self.a_bias.data)
                xavier_uniform_(self.r_bias.data)
            xavier_uniform_(self.relation_att.data)
            xavier_uniform_(self.relation_msg.data)
            xavier_uniform_(self.relation_pri.data)
        elif self.p.init_func == 'get_uniform':
            get_uniform(node_size, self.k_linear_h)
            get_uniform(node_size, self.q_linear_h)
            get_uniform(node_size, self.v_linear_h)
            get_uniform(node_size, self.k_linear_t)
            get_uniform(node_size, self.q_linear_t)
            get_uniform(node_size, self.v_linear_t)
            get_uniform(node_size, self.a_linear)
            get_uniform(node_size, self.r_linear)
            if self.p.bias != 0:
                get_uniform(node_size, sself.k_bias_h)
                get_uniform(node_size, sself.q_bias_h)
                get_uniform(node_size, sself.v_bias_h)
                get_uniform(node_size, sself.k_bias_t)
                get_uniform(node_size, sself.q_bias_t)
                get_uniform(node_size, sself.v_bias_t)
                get_uniform(node_size, sself.a_bias)
                get_uniform(node_size, sself.r_bias)
            get_uniform(re_size, self.relation_att)
            get_uniform(re_size, self.relation_msg)
            get_uniform(re_size, self.relation_pri)
        elif self.p.init_func == 'get_normal':
            get_normal(self.k_linear_h)
            get_normal(self.q_linear_h)
            get_normal(self.v_linear_h)
            get_normal(self.k_linear_t)
            get_normal(self.q_linear_t)
            get_normal(self.v_linear_t)
            get_normal(self.a_linear)
            get_normal(self.r_linear)
            if self.p.bias != 0:
                get_normal(self.k_bias_h)
                get_normal(self.q_bias_h)
                get_normal(self.v_bias_h)
                get_normal(self.k_bias_t)
                get_normal(self.q_bias_t)
                get_normal(self.v_bias_t)
                get_normal(self.a_bias)
                get_normal(self.r_bias)
            get_normal(self.relation_att)
            get_normal(self.relation_msg)
            get_normal(self.relation_pri)
        else:
            raise Exception('input one of xavier_normal/xavier_uniform/get_uniform/get_normal')

    def forward(self, node_embed, node_type, type_count, edge_index, edge_type, rel_embed, adj):
        loop_index = edge_index.size(1) - node_embed.size(0)
        num_edges = loop_index // 2
        num_ent = node_embed.size(0)
        self.in_index, self.out_index = (edge_index[:, :num_edges], edge_index[:, num_edges:loop_index])
        self.in_type, self.out_type = (edge_type[:num_edges], edge_type[num_edges:loop_index])
        self.loop_index = edge_index[:, loop_index:]
        self.loop_type = edge_type[loop_index:]
        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)
        r_ln = self.r_linear
        r = rel_embed
        bgn = time.time()
        in_res = self.propagate(edge_index=self.in_index, node_embed=node_embed, edge_type=self.in_type, rel_embed=rel_embed, node_type=node_type, type_count=type_count, edge_norm=self.in_norm, mode='in')
        end = time.time()
        bgn = end
        loop_res = self.propagate(edge_index=self.loop_index, node_embed=node_embed, edge_type=self.loop_type, rel_embed=rel_embed, node_type=node_type, type_count=type_count, edge_norm=None, mode='loop')
        end = time.time()
        bgn = end
        out_res = self.propagate(edge_index=self.out_index, node_embed=node_embed, edge_type=self.out_type, rel_embed=rel_embed, node_type=node_type, type_count=type_count, edge_norm=self.out_norm, mode='out')
        end = time.time()
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)
        return (out, self.drop(self.act_f(r_ln * r + self.r_bias)))

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        return norm

    def message(self, edge_index, edge_type, edge_index_i, edge_index_j, edge_norm, mode, rel_embed, node_embed, node_embed_j, node_type, node_type_j, type_count):
        data_size = edge_index_j.size(0)
        r_out = rel_embed
        rel_emb = torch.index_select(r_out, 0, edge_type)
        if mode == 'in':
            k_ln = self.k_linear_h
            q_ln = self.q_linear_h
            v_ln = self.v_linear_h
            x = node_embed
            k_x_t = k_ln[node_type.view(-1)]
            q_x_t = q_ln[node_type.view(-1)]
            v_x_t = v_ln[node_type.view(-1)]
            k_out = x * k_x_t + self.k_bias_h
            q_out = x * q_x_t + self.k_bias_h
            v_out = x * v_x_t + self.k_bias_h
        elif mode == 'out':
            k_ln = self.k_linear_t
            q_ln = self.q_linear_t
            v_ln = self.v_linear_t
            x = node_embed
            k_x_t = k_ln[node_type.view(-1)]
            q_x_t = q_ln[node_type.view(-1)]
            v_x_t = v_ln[node_type.view(-1)]
            k_out = x * k_x_t + self.k_bias_t
            q_out = x * q_x_t + self.k_bias_t
            v_out = x * v_x_t + self.k_bias_t
        elif mode == 'loop':
            x = node_embed
            k_out = q_out = v_out = self.loop(x)
        else:
            exit('mode error')
        k_m = torch.index_select(k_out, 0, edge_index.t()[:, 1])
        q_m = torch.index_select(q_out, 0, edge_index.t()[:, 0])
        v_m = torch.index_select(v_out, 0, edge_index.t()[:, 1])
        k_m = self.rel_transform(k_m, rel_emb)
        v_m = self.rel_transform(v_m, rel_emb)
        k_node_vec = k_m.view(-1, self.n_heads, self.d_k)
        q_node_vec = q_m.view(-1, self.n_heads, self.d_k)
        v_node_vec = v_m.view(-1, self.n_heads, self.d_k)
        k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[0]).transpose(1, 0)
        res_att = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[0] / self.sqrt_dk
        res_msg = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[0]).transpose(1, 0)
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        return res.reshape(-1, self.out_dim)

    def update(self, aggr_out, node_embed, node_type, type_count, mode):
        a_ln = self.a_linear
        x = aggr_out
        a_x_t = a_ln[node_type.view(-1)]
        a_m = x * a_x_t + self.a_bias
        if mode == 'in':
            if self.use_norm:
                a_m = self.n_norms[0](a_m)
        elif mode == 'loop':
            if self.use_norm:
                a_m = self.n_norms[1](a_m)
        elif mode == 'out':
            if self.use_norm:
                a_m = self.n_norms[2](a_m)
        res = self.act_f(a_m)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class old_LGTNv3Conv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(old_LGTNv3Conv, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        self.num_type_bases = 5
        self.norms = nn.ModuleList()
        for t in range(1):
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.k_bias = nn.Parameter(torch.Tensor(out_dim)).cuda()
            self.a_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.a_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            if self.p.bias != 0:
                self.a_bias = nn.Parameter(torch.Tensor(out_dim)).cuda()
        self.drop = nn.Dropout(dropout)
        self.hide_linears = nn.ModuleList()
        for t in range(self.p.n_linear):
            self.hide_linears.append(nn.Linear(out_dim * self.p.dimrate_linear, out_dim * self.p.dimrate_linear))
        self.in_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.out_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        if self.p.skip1:
            self.skip1 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.skip2:
            self.skip2 = nn.Parameter(torch.ones(self.num_maintype))
        self.relation_att = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_k, self.d_k)).cuda()
        self.relation_msg = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_k, self.d_k)).cuda()
        self.relation_pri = nn.Parameter(torch.Tensor(1, self.n_heads)).cuda()
        self.reset_parameters()
        self.rin_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.rout_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.k_l = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.v_l = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.q_l = nn.Linear(self.out_dim, self.out_dim, bias=False)

    def reset_parameters(self):
        node_size = self.p.node_num_bases * self.in_dim
        re_size = self.p.re_num_bases * self.n_heads
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.k_bias.data)
            xavier_normal_(self.a_basis.data)
            xavier_normal_(self.a_att.data)
            if self.p.bias != 0:
                xavier_normal_(self.a_bias.data)
            xavier_normal_(self.at_basis.data)
            xavier_normal_(self.at_att.data)
            xavier_normal_(self.ms_basis.data)
            xavier_normal_(self.ms_att.data)
            xavier_normal_(self.pr_basis.data)
            xavier_normal_(self.pr_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.k_bias)
            xavier_uniform_(self.a_basis.data)
            xavier_uniform_(self.a_att.data)
            if self.p.bias != 0:
                get_uniform(node_size, self.a_bias)
            xavier_uniform_(self.relation_att.data)
            xavier_uniform_(self.relation_msg.data)
            xavier_uniform_(self.relation_pri.data)
        elif self.p.init_func == 'get_uniform':
            get_uniform(node_size, self.k_basis)
            get_uniform(node_size, self.k_att)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            get_uniform(node_size, self.q_basis)
            get_uniform(node_size, self.q_att)
            get_uniform(node_size, self.q_bias)
            get_uniform(node_size, self.q_bias_att)
            get_uniform(node_size, self.v_basis)
            get_uniform(node_size, self.v_att)
            get_uniform(node_size, self.v_bias)
            get_uniform(node_size, self.v_bias_att)
            get_uniform(node_size, self.a_basis)
            get_uniform(node_size, self.a_att)
            get_uniform(node_size, self.a_bias)
            get_uniform(node_size, self.a_bias_att)
            get_uniform(re_size, self.at_basis)
            get_uniform(re_size, self.at_att)
            get_uniform(re_size, self.ms_basis)
            get_uniform(re_size, self.ms_att)
            get_uniform(re_size, self.pr_basis)
            get_uniform(re_size, self.pr_att)
        elif self.p.init_func == 'get_normal':
            get_normal(self.k_basis)
            get_normal(self.k_att)
            get_normal(self.k_bias)
            get_normal(self.k_bias_att)
            get_normal(self.q_basis)
            get_normal(self.q_att)
            get_normal(self.q_bias)
            get_normal(self.q_bias_att)
            get_normal(self.v_basis)
            get_normal(self.v_att)
            get_normal(self.v_bias)
            get_normal(self.v_bias_att)
            get_normal(self.a_basis)
            get_normal(self.a_att)
            get_normal(self.a_bias)
            get_normal(self.a_bias_att)
            get_normal(self.at_basis)
            get_normal(self.at_att)
            get_normal(self.ms_basis)
            get_normal(self.ms_att)
            get_normal(self.pr_basis)
            get_normal(self.pr_att)
        else:
            raise Exception('input one of xavier_normal/xavier_uniform/get_uniform/get_normal')

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, rel_inp, adj):
        meta_xs = self.propagate(edge_index, node_inp=node_inp, node_maintype=node_maintype, type_count=type_count, edge_type=edge_type, rel_inp=rel_inp)
        meta_rel = self.rel_process(rel_inp, edge_index)
        return (meta_xs, meta_rel)

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def message(self, edge_index_i, edge_index, edge_type, node_inp, node_maintype, type_count, rel_inp):
        start = time.time()
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp.device)
        length = node_maintype.size(1)
        k_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.out_dim)
        if self.p.bias != 0:
            k_bs = self.k_bias
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    k_b = k_bs
                else:
                    k_b = 0
                k_m = node_inp * k_w + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    k_b = k_bs
                else:
                    k_b = 0
                k_m = node_inp * k_w + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        k_m = torch.index_select(k_sum, 0, edge_index.t()[:, 1])
        v_m = torch.index_select(k_sum, 0, edge_index.t()[:, 1])
        q_m = torch.index_select(k_sum, 0, edge_index.t()[:, 0])
        rel_emb = torch.index_select(rel_inp, 0, edge_type)
        k_m = self.rel_transform(k_m, rel_emb)
        v_m = self.rel_transform(v_m, rel_emb)
        q_m = self.rel_transform(q_m, rel_emb)
        k_m = self.k_l(k_m)
        v_m = self.v_l(v_m)
        q_m = self.q_l(q_m)
        k_node_vec = k_m.view(-1, self.n_heads, self.d_k)
        v_node_vec = v_m.view(-1, self.n_heads, self.d_k)
        q_node_vec = q_m.view(-1, self.n_heads, self.d_k)
        k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[0]).transpose(1, 0)
        res_att = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[0] / self.sqrt_dk
        res_msg = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[0]).transpose(1, 0)
        del k_m
        del v_m
        del q_m
        gc.collect()
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        gc.collect()
        end = time.time()
        return res.reshape(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_maintype, type_count):
        start = time.time()
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        length = node_maintype.size(1)
        a_sum = 0
        a_ws = torch.matmul(self.a_att, self.a_basis.view(self.p.node_num_bases, -1))
        a_ws = a_ws.view(self.num_maintype, self.out_dim)
        if self.p.bias != 0:
            a_bs = self.a_bias
        if self.p.type_mode == 'multiple':
            for i in range(length):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    a_b = a_bs
                else:
                    a_b = 0
                a_m = aggr_out * a_w + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                if self.p.bias != 0:
                    a_b = a_bs
                else:
                    a_b = 0
                a_m = aggr_out * a_w + a_b
                a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        a_m = self.drop(a_sum)
        if self.p.skip1:
            alpha1 = torch.index_select(self.skip1, 0, node_maintype).view(-1, 1)
        if self.p.skip2:
            alpha2 = torch.index_select(self.skip2, 0, node_maintype).view(-1, 1)
        if self.p.skip1:
            a_m = a_m * alpha1 + node_inp * (1 - alpha1)
        else:
            a_m = a_m + node_inp
        if self.use_norm:
            a_m = self.norms[0](a_m)
        mat = F.gelu(self.in_linear(a_m))
        for line in range(self.p.n_linear):
            mat = self.drop(F.gelu(self.hide_linears[line](mat)))
        mat = self.drop(self.out_linear(mat))
        if self.p.skip2:
            mat = mat * alpha2 + a_m * (1 - alpha2)
        else:
            mat = mat + a_m
        res = self.out_norm(mat)
        return res

    def rel_process(self, rel_inp, edge_index):
        rk_node_vec = rel_inp.view(-1, self.n_heads, self.d_k)
        rk_node_vec = torch.bmm(rk_node_vec.transpose(1, 0), self.relation_att[0]).transpose(1, 0)
        rel = rk_node_vec.contiguous().view(self.num_relations, -1)
        start = time.time()
        ra_m = F.gelu(rel)
        ra_m = self.drop(ra_m)
        if self.p.skip1:
            ra_m = ra_m * alpha1 + rel_inp * (1 - alpha1)
        else:
            ra_m = ra_m + rel_inp
        rmat = F.gelu(self.rin_linear(ra_m))
        for line in range(self.p.n_linear):
            rmat = self.drop(F.gelu(self.rhide_linears[line](rmat)))
        rmat = self.drop(self.rout_linear(rmat))
        if self.p.skip2:
            rmat = rmat * alpha2 + ra_m * (1 - alpha2)
        else:
            rmat = rmat + ra_m
        rel = rmat
        return rel

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class NoRelDenseMTHGCLConv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(NoRelDenseMTHGCLConv, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        self.num_type_bases = 5
        self.norms = nn.ModuleList()
        for t in range(1):
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        if self.p.node_num_bases > 0 and self.p.model_nodetype == 1:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.v_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.v_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.v_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.v_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.q_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.q_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.q_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.q_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.a_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim, out_dim)).cuda()
            self.a_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.a_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.a_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
        else:
            self.k_linear = nn.Linear(self.in_dim, self.out_dim)
            self.v_linear = nn.Linear(self.in_dim, self.out_dim)
            self.q_linear = nn.Linear(self.in_dim, self.out_dim)
            self.a_linear = nn.Linear(self.out_dim, self.out_dim)
        self.drop = nn.Dropout(dropout)
        self.hide_linears = nn.ModuleList()
        for t in range(self.p.n_linear):
            self.hide_linears.append(nn.Linear(out_dim * self.p.dimrate_linear, out_dim * self.p.dimrate_linear))
        self.in_linear = nn.Linear(out_dim, out_dim * self.p.dimrate_linear)
        self.out_linear = nn.Linear(out_dim * self.p.dimrate_linear, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        if self.p.skip1:
            self.skip1 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.skip2:
            self.skip2 = nn.Parameter(torch.ones(self.num_maintype))
        if self.p.re_num_bases > 0 and self.p.model_relationtype == 1:
            self.at_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.at_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.ms_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads, self.d_k, self.d_k)).cuda()
            self.ms_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
            self.pr_basis = nn.Parameter(torch.Tensor(self.p.re_num_bases, self.n_heads)).cuda()
            self.pr_att = nn.Parameter(torch.Tensor(self.num_relations, self.p.re_num_bases)).cuda()
        else:
            self.relation_att = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_k, self.d_k))
            self.relation_msg = nn.Parameter(torch.Tensor(1, self.n_heads, self.d_k, self.d_k))
            self.relation_pri = nn.Parameter(torch.Tensor(1, self.n_heads))
        self.reset_parameters()
        if self.p.model_relationtype == 1:
            self.relation_att = torch.matmul(self.at_att, self.at_basis.view(self.p.re_num_bases, -1))
            self.relation_att = self.relation_att.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
            self.relation_msg = torch.matmul(self.ms_att, self.ms_basis.view(self.p.re_num_bases, -1))
            self.relation_msg = self.relation_msg.view(self.num_relations, self.n_heads, self.d_k, self.d_k)
            self.relation_pri = torch.matmul(self.pr_att, self.pr_basis)

    def reset_parameters(self):
        node_size = self.p.node_num_bases * self.in_dim
        re_size = self.p.re_num_bases * self.n_heads
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
            xavier_normal_(self.q_basis.data)
            xavier_normal_(self.q_att.data)
            xavier_normal_(self.q_bias.data)
            xavier_normal_(self.q_bias_att.data)
            xavier_normal_(self.v_basis.data)
            xavier_normal_(self.v_att.data)
            xavier_normal_(self.v_bias.data)
            xavier_normal_(self.v_bias_att.data)
            xavier_normal_(self.a_basis.data)
            xavier_normal_(self.a_att.data)
            xavier_normal_(self.a_bias.data)
            xavier_normal_(self.a_bias_att.data)
            xavier_normal_(self.at_basis.data)
            xavier_normal_(self.at_att.data)
            xavier_normal_(self.ms_basis.data)
            xavier_normal_(self.ms_att.data)
            xavier_normal_(self.pr_basis.data)
            xavier_normal_(self.pr_att.data)
        elif self.p.init_func == 'xavier_uniform':
            if self.p.model_nodetype == 1:
                xavier_uniform_(self.k_basis.data)
                xavier_uniform_(self.k_att.data)
                get_uniform(node_size, self.k_bias)
                get_uniform(node_size, self.k_bias_att)
                xavier_uniform_(self.q_basis.data)
                xavier_uniform_(self.q_att.data)
                get_uniform(node_size, self.q_bias)
                get_uniform(node_size, self.q_bias_att)
                xavier_uniform_(self.v_basis.data)
                xavier_uniform_(self.v_att.data)
                get_uniform(node_size, self.v_bias)
                get_uniform(node_size, self.v_bias_att)
                xavier_uniform_(self.a_basis.data)
                xavier_uniform_(self.a_att.data)
                get_uniform(node_size, self.a_bias)
                get_uniform(node_size, self.a_bias_att)
            if self.p.model_relationtype == 1:
                xavier_uniform_(self.at_basis.data)
                xavier_uniform_(self.at_att.data)
                xavier_uniform_(self.ms_basis.data)
                xavier_uniform_(self.ms_att.data)
                xavier_uniform_(self.pr_basis.data)
                xavier_uniform_(self.pr_att.data)
            else:
                xavier_uniform_(self.relation_att.data)
                xavier_uniform_(self.relation_msg.data)
                xavier_uniform_(self.relation_pri.data)
        elif self.p.init_func == 'get_uniform':
            get_uniform(node_size, self.k_basis)
            get_uniform(node_size, self.k_att)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)
            get_uniform(node_size, self.q_basis)
            get_uniform(node_size, self.q_att)
            get_uniform(node_size, self.q_bias)
            get_uniform(node_size, self.q_bias_att)
            get_uniform(node_size, self.v_basis)
            get_uniform(node_size, self.v_att)
            get_uniform(node_size, self.v_bias)
            get_uniform(node_size, self.v_bias_att)
            get_uniform(node_size, self.a_basis)
            get_uniform(node_size, self.a_att)
            get_uniform(node_size, self.a_bias)
            get_uniform(node_size, self.a_bias_att)
            get_uniform(re_size, self.at_basis)
            get_uniform(re_size, self.at_att)
            get_uniform(re_size, self.ms_basis)
            get_uniform(re_size, self.ms_att)
            get_uniform(re_size, self.pr_basis)
            get_uniform(re_size, self.pr_att)
        elif self.p.init_func == 'get_normal':
            get_normal(self.k_basis)
            get_normal(self.k_att)
            get_normal(self.k_bias)
            get_normal(self.k_bias_att)
            get_normal(self.q_basis)
            get_normal(self.q_att)
            get_normal(self.q_bias)
            get_normal(self.q_bias_att)
            get_normal(self.v_basis)
            get_normal(self.v_att)
            get_normal(self.v_bias)
            get_normal(self.v_bias_att)
            get_normal(self.a_basis)
            get_normal(self.a_att)
            get_normal(self.a_bias)
            get_normal(self.a_bias_att)
            get_normal(self.at_basis)
            get_normal(self.at_att)
            get_normal(self.ms_basis)
            get_normal(self.ms_att)
            get_normal(self.pr_basis)
            get_normal(self.pr_att)
        else:
            raise Exception('input one of xavier_normal/xavier_uniform/get_uniform/get_normal')

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, r, adj):
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        rs = self.propagate(edge_index, node_inp=node_inp, node_maintype=node_maintype, type_count=type_count, edge_type=edge_type)
        return rs

    def message(self, edge_index_i, edge_index, edge_type, node_inp, node_maintype, type_count):
        start = time.time()
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp.device)
        length = node_maintype.size(1)
        k_sum = 0
        v_sum = 0
        q_sum = 0
        if self.p.model_nodetype == 1:
            k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
            k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
            k_bs = torch.matmul(self.k_bias_att, self.k_bias)
            v_ws = torch.matmul(self.v_att, self.v_basis.view(self.p.node_num_bases, -1))
            v_ws = v_ws.view(self.num_maintype, self.in_dim, self.out_dim)
            v_bs = torch.matmul(self.v_bias_att, self.v_bias)
            q_ws = torch.matmul(self.q_att, self.q_basis.view(self.p.node_num_bases, -1))
            q_ws = q_ws.view(self.num_maintype, self.in_dim, self.out_dim)
            q_bs = torch.matmul(self.q_bias_att, self.q_bias)
            if self.p.type_mode == 'multiple':
                for i in range(length):
                    k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                    k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                    if self.p.att == 'adaptive_att':
                        k_sum = k_sum + k_m * torch.softmax(type_count[:, i], dim=-1).view(-1, 1)
                    else:
                        k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
                    v_w = torch.index_select(v_ws, 0, node_maintype[:, i])
                    v_b = torch.index_select(v_bs, 0, node_maintype[:, i])
                    v_m = torch.bmm(node_inp.unsqueeze(1), v_w).squeeze(-2) + v_b
                    if self.p.att == 'adaptive_att':
                        v_sum = v_sum + v_m * torch.softmax(type_count[:, i], dim=-1).view(-1, 1)
                    else:
                        v_sum = v_sum + v_m * type_count[:, i].view(-1, 1)
                    q_w = torch.index_select(q_ws, 0, node_maintype[:, i])
                    q_b = torch.index_select(q_bs, 0, node_maintype[:, i])
                    q_m = torch.bmm(node_inp.unsqueeze(1), q_w).squeeze(-2) + q_b
                    if self.p.att == 'adaptive_att':
                        q_sum = q_sum + q_m * torch.softmax(type_count[:, i], dim=-1).view(-1, 1)
                    else:
                        q_sum = q_sum + q_m * type_count[:, i].view(-1, 1)
            elif self.p.type_mode == 'single':
                for i in range(1):
                    k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                    k_m = torch.bmm(node_inp.unsqueeze(1), k_w).squeeze(-2) + k_b
                    if self.p.att == 'adaptive_att':
                        k_sum = k_sum + k_m * torch.softmax(type_count[:, i], dim=-1).view(-1, 1)
                    else:
                        k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
                    v_w = torch.index_select(v_ws, 0, node_maintype[:, i])
                    v_b = torch.index_select(v_bs, 0, node_maintype[:, i])
                    v_m = torch.bmm(node_inp.unsqueeze(1), v_w).squeeze(-2) + v_b
                    if self.p.att == 'adaptive_att':
                        v_sum = v_sum + v_m * torch.softmax(type_count[:, i], dim=-1).view(-1, 1)
                    else:
                        v_sum = v_sum + v_m * type_count[:, i].view(-1, 1)
                    q_w = torch.index_select(q_ws, 0, node_maintype[:, i])
                    q_b = torch.index_select(q_bs, 0, node_maintype[:, i])
                    q_m = torch.bmm(node_inp.unsqueeze(1), q_w).squeeze(-2) + q_b
                    if self.p.att == 'adaptive_att':
                        q_sum = q_sum + q_m * torch.softmax(type_count[:, i], dim=-1).view(-1, 1)
                    else:
                        q_sum = q_sum + q_m * type_count[:, i].view(-1, 1)
            else:
                raise Exception('input multiple/single')
        else:
            k_sum = self.k_linear(node_inp)
            v_sum = self.v_linear(node_inp)
            q_sum = self.q_linear(node_inp)
        k_m = torch.index_select(k_sum, 0, edge_index.t()[:, 1])
        v_m = torch.index_select(v_sum, 0, edge_index.t()[:, 1])
        q_m = torch.index_select(q_sum, 0, edge_index.t()[:, 0])
        for relation in range(self.num_relations):
            idx = edge_type == int(relation)
            if idx.sum() == 0:
                continue
            k_node_vec = k_m[idx].view(-1, self.n_heads, self.d_k)
            v_node_vec = v_m[idx].view(-1, self.n_heads, self.d_k)
            q_node_vec = q_m[idx].view(-1, self.n_heads, self.d_k)
            if self.p.model_relationtype == 1:
                k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[relation]).transpose(1, 0)
                res_att[idx] = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[relation] / self.sqrt_dk
                res_msg[idx] = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[relation]).transpose(1, 0)
            else:
                k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[0]).transpose(1, 0)
                res_att[idx] = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[0] / self.sqrt_dk
                res_msg[idx] = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[0]).transpose(1, 0)
        del k_m
        del v_m
        del q_m
        gc.collect()
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        gc.collect()
        end = time.time()
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_maintype, type_count):
        start = time.time()
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        length = node_maintype.size(1)
        a_sum = 0
        if self.p.model_nodetype == 1:
            a_ws = torch.matmul(self.a_att, self.a_basis.view(self.p.node_num_bases, -1))
            a_ws = a_ws.view(self.num_maintype, self.out_dim, self.out_dim)
            a_bs = torch.matmul(self.a_bias_att, self.a_bias)
            if self.p.type_mode == 'multiple':
                for i in range(length):
                    a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                    a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                    a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                    if self.p.att == 'adaptive_att':
                        a_sum = a_sum + a_m * torch.softmax(type_count[:, i], dim=-1).view(-1, 1)
                    else:
                        a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
            elif self.p.type_mode == 'single':
                for i in range(1):
                    a_w = torch.index_select(a_ws, 0, node_maintype[:, i])
                    a_b = torch.index_select(a_bs, 0, node_maintype[:, i])
                    a_m = torch.bmm(aggr_out.unsqueeze(1), a_w).squeeze(-2) + a_b
                    if self.p.att == 'adaptive_att':
                        a_sum = a_sum + a_m * torch.softmax(type_count[:, i], dim=-1).view(-1, 1)
                    else:
                        a_sum = a_sum + a_m * type_count[:, i].view(-1, 1)
            else:
                raise Exception('input multiple/single')
        else:
            a_sum = self.a_linear(aggr_out)
        a_m = self.drop(a_sum)
        if self.p.skip1:
            alpha1 = torch.index_select(self.skip1, 0, node_maintype).view(-1, 1)
        if self.p.skip2:
            alpha2 = torch.index_select(self.skip2, 0, node_maintype).view(-1, 1)
        if self.p.skip1:
            a_m = a_m * alpha1 + node_inp * (1 - alpha1)
        else:
            a_m = a_m + node_inp
        if self.use_norm:
            a_m = self.norms[0](a_m)
        mat = F.gelu(self.in_linear(a_m))
        for line in range(self.p.n_linear):
            mat = self.drop(F.gelu(self.hide_linears[line](mat)))
        mat = self.drop(self.out_linear(mat))
        if self.p.skip2:
            mat = mat * alpha2 + a_m * (1 - alpha2)
        else:
            mat = mat + a_m
        res = self.out_norm(mat)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class COMPConv(MessagePassingSelf):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(COMPConv, self).__init__(aggr='add', **kwargs)
        self.p = params
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_rels = (num_rel - 1) // 2
        self.act = torch.tanh
        self.device = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        self.w_loop = get_param((self.in_channels, self.out_channels))
        self.w_in = get_param((self.in_channels, self.out_channels))
        self.w_out = get_param((self.in_channels, self.out_channels))
        self.w_rel = get_param((self.in_channels, self.out_channels))
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(self.out_channels)
        self.comp_drop = torch.nn.Dropout(self.p.hid_drop)
        self.comp_feature_drop = torch.nn.Dropout(self.p.feat_drop)
        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(self.out_channels)))

    def reset_parameters(self):
        node_size = self.p.node_num_bases * self.in_dim
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj):
        if self.device is None:
            self.device = edge_index.device
        loop_index = edge_index.size(1) - x.size(0)
        num_edges = loop_index // 2
        num_ent = x.size(0)
        self.in_index, self.out_index = (edge_index[:, :num_edges], edge_index[:, num_edges:loop_index])
        self.in_type, self.out_type = (edge_type[:num_edges], edge_type[num_edges:loop_index])
        self.loop_index = edge_index[:, loop_index:]
        self.loop_type = edge_type[loop_index:]
        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)
        in_res = self.propagate('add', edge_index=self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count, edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate('add', edge_index=self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count, edge_norm=None, mode='loop')
        out_res = self.propagate('add', edge_index=self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count, edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)
        if self.p.bias:
            out = out + self.bias
        out = self.bn(out)
        return (self.comp_drop(self.act(out)), torch.matmul(rel_embed, self.w_rel))

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode, edge_index, x, node_maintype, type_count):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        return norm

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels * 2 + 1)

class old_COMPConv(MessagePassingSelf):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(COMPConv, self).__init__(aggr='add', **kwargs)
        self.p = params
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_rels = (num_rel - 1) // 2
        self.act = torch.tanh
        self.device = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        self.w_loop = get_param((self.in_channels, self.out_channels))
        self.w_in = get_param((self.in_channels, self.out_channels))
        self.w_out = get_param((self.in_channels, self.out_channels))
        self.w_rel = get_param((self.in_channels, self.out_channels))
        self.loop_rel = get_param((1, self.in_channels))
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(self.out_channels)
        self.comp_drop = torch.nn.Dropout(self.p.hid_drop)
        self.comp_feature_drop = torch.nn.Dropout(self.p.feat_drop)
        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(self.out_channels)))

    def reset_parameters(self):
        node_size = self.p.node_num_bases * self.in_dim
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj):
        if self.device is None:
            self.device = edge_index.device
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)
        self.in_index, self.out_index = (edge_index[:, :num_edges], edge_index[:, num_edges:])
        self.in_type, self.out_type = (edge_type[:num_edges], edge_type[num_edges:])
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)
        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)
        k_x = x
        in_res = self.propagate('add', edge_index=self.in_index, x=k_x, edge_type=self.in_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count, edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate('add', edge_index=self.loop_index, x=k_x, edge_type=self.loop_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count, edge_norm=None, mode='loop')
        out_res = self.propagate('add', edge_index=self.out_index, x=k_x, edge_type=self.out_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count, edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)
        if self.p.bias:
            out = out + self.bias
        out = self.bn(out)
        return (self.comp_drop(self.act(out)), torch.matmul(rel_embed, self.w_rel)[:-1])

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode, edge_index, x, node_maintype, type_count):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

class copy_COMPConv(MessagePassingSelf):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(copy_COMPConv, self).__init__(aggr='add', **kwargs)
        self.p = params
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_rels = num_rel - 1
        self.act = torch.tanh
        self.device = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        if self.p.node_num_bases > 0:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.reset_parameters()
        self.w_loop = get_param((self.in_channels, self.out_channels))
        self.w_in = get_param((self.in_channels, self.out_channels))
        self.w_out = get_param((self.in_channels, self.out_channels))
        self.w_rel = get_param((self.in_channels, self.out_channels))
        self.loop_rel = get_param((1, self.in_channels))
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(self.out_channels)
        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(self.out_channels)))

    def reset_parameters(self):
        node_size = self.p.node_num_bases * self.in_dim
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj):
        if self.device is None:
            self.device = edge_index.device
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)
        self.in_index, self.out_index = (edge_index[:, :num_edges], edge_index[:, num_edges:])
        self.in_type, self.out_type = (edge_type[:num_edges], edge_type[num_edges:])
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)
        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)
        length = node_type.size(1)
        node_maintype = node_type
        k_sum = 0
        k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
        k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
        k_bs = torch.matmul(self.k_bias_att, self.k_bias)
        if self.p.type_mode == 'multiple':
            for i in range(length):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        elif self.p.type_mode == 'single':
            for i in range(1):
                k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
        else:
            raise Exception('input multiple/single')
        k_x = k_sum
        in_res = self.propagate('add', edge_index=self.in_index, x=k_x, edge_type=self.in_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count, edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate('add', edge_index=self.loop_index, x=k_x, edge_type=self.loop_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count, edge_norm=None, mode='loop')
        out_res = self.propagate('add', edge_index=self.out_index, x=k_x, edge_type=self.out_type, rel_embed=rel_embed, node_maintype=node_type, type_count=type_count, edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)
        if self.p.bias:
            out = out + self.bias
        out = self.bn(out)
        return (self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1])

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode, edge_index, x, node_maintype, type_count):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

class SAGE(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, bias=True, **kwargs):
        super(SAGE, self).__init__(aggr='add', **kwargs)
        self.in_features = in_dim
        self.out_features = out_dim
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.num_relations = num_rel
        self.alpha = torch.nn.Embedding(num_rel, 1, padding_idx=0)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        if self.p.node_num_bases > 0 and self.p.model_nodetype == 1:
            self.k_basis = nn.Parameter(torch.Tensor(self.p.node_num_bases, in_dim, out_dim)).cuda()
            self.k_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.k_bias = nn.Parameter(torch.Tensor(self.p.node_num_bases, out_dim)).cuda()
            self.k_bias_att = nn.Parameter(torch.Tensor(self.num_maintype, self.p.node_num_bases)).cuda()
            self.reset_parameters2()
        else:
            self.adapt_ws = nn.Linear(self.in_dim, self.out_dim)

    def reset_parameters2(self):
        node_size = self.p.node_num_bases * self.in_dim
        if self.p.init_func == 'xavier_normal':
            xavier_normal_(self.k_basis.data)
            xavier_normal_(self.k_att.data)
            xavier_normal_(self.k_bias.data)
            xavier_normal_(self.k_bias_att.data)
        elif self.p.init_func == 'xavier_uniform':
            xavier_uniform_(self.k_basis.data)
            xavier_uniform_(self.k_att.data)
            get_uniform(node_size, self.k_bias)
            get_uniform(node_size, self.k_bias_att)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj):
        if self.p.model_nodetype == 1:
            node_maintype = node_type
            length = node_maintype.size(1)
            k_sum = 0
            k_ws = torch.matmul(self.k_att, self.k_basis.view(self.p.node_num_bases, -1))
            k_ws = k_ws.view(self.num_maintype, self.in_dim, self.out_dim)
            k_bs = torch.matmul(self.k_bias_att, self.k_bias)
            if self.p.type_mode == 'multiple':
                for i in range(length):
                    k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                    k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                    k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
            elif self.p.type_mode == 'single':
                for i in range(1):
                    k_w = torch.index_select(k_ws, 0, node_maintype[:, i])
                    k_b = torch.index_select(k_bs, 0, node_maintype[:, i])
                    k_m = torch.bmm(x.unsqueeze(1), k_w).squeeze(-2) + k_b
                    k_sum = k_sum + k_m * type_count[:, i].view(-1, 1)
            else:
                raise Exception('input multiple/single')
        else:
            k_sum = self.adapt_ws(x)
        alp = self.alpha(adj[1]).t()[0]
        A = torch.sparse_coo_tensor(adj[0], alp, torch.Size([adj[2], adj[2]]), requires_grad=True)
        A = A + A.transpose(0, 1)
        support = torch.mm(k_sum, self.weight)
        output = torch.sparse.mm(A, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Origin_COMPConv(MessagePassingSelf):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(Origin_COMPConv, self).__init__(aggr='add', **kwargs)
        self.p = params
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_rels = num_rel - 1
        self.act = torch.tanh
        self.device = None
        self.w_loop = get_param((self.in_channels, self.out_channels))
        self.w_in = get_param((self.in_channels, self.out_channels))
        self.w_out = get_param((self.in_channels, self.out_channels))
        self.w_rel = get_param((self.in_channels, self.out_channels))
        self.loop_rel = get_param((1, self.in_channels))
        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(self.out_channels)
        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(self.out_channels)))

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj):
        if self.device is None:
            self.device = edge_index.device
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)
        self.in_index, self.out_index = (edge_index[:, :num_edges], edge_index[:, num_edges:])
        self.in_type, self.out_type = (edge_type[:num_edges], edge_type[num_edges:])
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)
        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)
        in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed, edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, mode='loop')
        out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed, edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)
        if self.p.bias:
            out = out + self.bias
        out = self.bn(out)
        return (self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1])

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError
        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)
        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]
        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

class OriginSAGE(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, bias=True, **kwargs):
        super(OriginSAGE, self).__init__(aggr='add', **kwargs)
        self.in_features = in_dim
        self.out_features = out_dim
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.num_relations = num_rel
        self.alpha = torch.nn.Embedding(num_rel, 1, padding_idx=0)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, node_type, type_count, edge_index, edge_type, rel_embed, adj):
        alp = self.alpha(adj[1]).t()[0]
        A = torch.sparse_coo_tensor(adj[0], alp, torch.Size([adj[2], adj[2]]), requires_grad=True)
        A = A + A.transpose(0, 1)
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(A, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class Origin_HGTConv(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(Origin_HGTConv, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        self.num_type_bases = 5
        self.norms = nn.ModuleList()
        for t in range(self.num_maintype):
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        if self.p.node_num_bases > 0:
            self.k_linears = []
            self.q_linears = []
            self.v_linears = []
            self.a_linears = []
            self.k_linears_bias = []
            self.q_linears_bias = []
            self.v_linears_bias = []
            self.a_linears_bias = []
            for r in range(self.p.node_num_bases):
                self.k_linears.append(get_param((in_dim, out_dim)).view(-1))
                self.q_linears.append(get_param((in_dim, out_dim)).view(-1))
                self.v_linears.append(get_param((in_dim, out_dim)).view(-1))
                self.a_linears.append(get_param((out_dim, out_dim)).view(-1))
                self.k_linears_bias.append(Parameter(torch.zeros(out_dim)))
                self.q_linears_bias.append(Parameter(torch.zeros(out_dim)))
                self.v_linears_bias.append(Parameter(torch.zeros(out_dim)))
                self.a_linears_bias.append(Parameter(torch.zeros(out_dim)))
            self.k_linears = torch.stack(self.k_linears, dim=0)
            self.q_linears = torch.stack(self.q_linears, dim=0)
            self.v_linears = torch.stack(self.v_linears, dim=0)
            self.a_linears = torch.stack(self.a_linears, dim=0)
            self.k_linears_bias = torch.stack(self.k_linears_bias, dim=0)
            self.q_linears_bias = torch.stack(self.q_linears_bias, dim=0)
            self.v_linears_bias = torch.stack(self.v_linears_bias, dim=0)
            self.a_linears_bias = torch.stack(self.a_linears_bias, dim=0)
            k_type_w = get_param((self.num_maintype, self.p.node_num_bases))
            q_type_w = get_param((self.num_maintype, self.p.node_num_bases))
            v_type_w = get_param((self.num_maintype, self.p.node_num_bases))
            a_type_w = get_param((self.num_maintype, self.p.node_num_bases))
            self.k_linears = k_type_w @ self.k_linears
            self.q_linears = q_type_w @ self.q_linears
            self.v_linears = v_type_w @ self.v_linears
            self.a_linears = a_type_w @ self.a_linears
            self.k_linears_bias = k_type_w @ self.k_linears_bias
            self.q_linears_bias = q_type_w @ self.q_linears_bias
            self.v_linears_bias = v_type_w @ self.v_linears_bias
            self.a_linears_bias = a_type_w @ self.a_linears_bias
            self.k_linears = self.k_linears.view(self.num_maintype, in_dim, out_dim).cuda()
            self.q_linears = self.q_linears.view(self.num_maintype, in_dim, out_dim).cuda()
            self.v_linears = self.v_linears.view(self.num_maintype, in_dim, out_dim).cuda()
            self.a_linears = self.a_linears.view(self.num_maintype, out_dim, out_dim).cuda()
            self.k_linears_bias = self.k_linears_bias.cuda()
            self.q_linears_bias = self.q_linears_bias.cuda()
            self.v_linears_bias = self.v_linears_bias.cuda()
            self.a_linears_bias = self.a_linears_bias.cuda()
        self.skip = nn.Parameter(torch.ones(self.num_maintype))
        self.drop = nn.Dropout(dropout)
        if self.p.re_num_bases > 0:
            self.relation_att = []
            self.relation_msg = []
            for r in range(self.re_num_bases):
                self.relation_att.append(get_param((self.n_heads, self.d_k, self.d_k)).view(-1))
                self.relation_msg.append(get_param((self.n_heads, self.d_k, self.d_k)).view(-1))
            self.relation_att = torch.stack(self.relation_att, dim=0)
            self.relation_msg = torch.stack(self.relation_msg, dim=0)
            att_type_w = get_param((self.num_relations, self.re_num_bases))
            msg_type_w = get_param((self.num_relations, self.re_num_bases))
            self.relation_att = att_type_w @ self.relation_att
            self.relation_msg = msg_type_w @ self.relation_msg
            self.relation_att = self.relation_att.view(self.num_relations, self.n_heads, self.d_k, self.d_k).cuda()
            self.relation_msg = self.relation_msg.view(self.num_relations, self.n_heads, self.d_k, self.d_k).cuda()
            self.relation_pri = []
            for r in range(self.re_num_bases):
                self.relation_pri.append(nn.Parameter(torch.ones(self.n_heads)))
            self.relation_pri = torch.stack(self.relation_pri, dim=0)
            pri_type_w = get_param((self.num_relations, self.re_num_bases))
            self.relation_pri = pri_type_w @ self.relation_pri
            self.relation_pri = self.relation_pri.cuda()

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, r):
        self.node_num_bases = self.p.node_num_bases
        self.re_num_bases = self.p.re_num_bases
        rs = self.propagate(edge_index, node_inp=node_inp, node_maintype=node_maintype, edge_type=edge_type)
        return rs

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_maintype_i, node_maintype_j, edge_index, edge_type, node_inp, node_maintype):
        start = time.time()
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        k_linears_dict = {}
        v_linears_dict = {}
        q_linears_dict = {}
        k_m = []
        v_m = []
        q_m = []
        k = OrderedSet()
        v = OrderedSet()
        q = OrderedSet()
        for edge in edge_index.t():
            h = int(edge[0])
            t = int(edge[1])
            k.add(t)
            v.add(t)
            q.add(h)
        if len(k) != len(q):
            print('len(k) != len(q)')
            exit()
        for k_id, v_id, q_id in zip(k, v, q):
            node_inp_t = node_inp[k_id]
            node_inp_h = node_inp[q_id]
            types_t_id = node_maintype[k_id]
            types_h_id = node_maintype[q_id]
            types_t = self.p.id2types[int(types_t_id)]
            types_h = self.p.id2types[int(types_h_id)]
            node_t_id = k_id
            node_h_id = q_id
            k_mat = []
            k_att = self.p.types2count[types_t]
            k_att = torch.softmax(torch.FloatTensor(k_att), dim=-1).to(node_inp_i.device)
            for stype in types_t.split(' '):
                stype_id = self.p.type2id[stype]
                w = self.k_linears[stype_id]
                bias = self.k_linears_bias[stype_id]
                mat = node_inp_t @ w + bias
                k_mat.append(mat)
            k_mat = torch.stack(k_mat, 0)
            k_att = k_att.unsqueeze(-1)
            k_mat = k_att * k_mat
            k_mat = k_mat.sum(dim=0)
            k_mat = k_mat.view(-1, self.n_heads, self.d_k)
            k_linears_dict[node_t_id] = k_mat
            v_mat = []
            v_att = self.p.types2count[types_t]
            v_att = torch.softmax(torch.FloatTensor(v_att), dim=-1).to(node_inp_i.device)
            for stype in types_t.split(' '):
                stype_id = self.p.type2id[stype]
                w = self.v_linears[stype_id]
                bias = self.v_linears_bias[stype_id]
                mat = node_inp_t @ w + bias
                v_mat.append(mat)
            v_mat = torch.stack(v_mat, 0)
            v_att = v_att.unsqueeze(-1)
            v_mat = v_att * v_mat
            v_mat = v_mat.sum(dim=0)
            v_mat = v_mat.view(-1, self.n_heads, self.d_k)
            v_linears_dict[node_t_id] = v_mat
            q_mat = []
            q_att = self.p.types2count[types_h]
            q_att = torch.softmax(torch.FloatTensor(q_att), dim=-1).to(node_inp_i.device)
            for stype in types_h.split(' '):
                stype_id = self.p.type2id[stype]
                w = self.q_linears[stype_id]
                bias = self.q_linears_bias[stype_id]
                mat = node_inp_h @ w + bias
                q_mat.append(mat)
            q_mat = torch.stack(q_mat, 0)
            q_att = q_att.unsqueeze(-1)
            q_mat = q_att * q_mat
            q_mat = q_mat.sum(dim=0)
            q_mat = q_mat.view(-1, self.n_heads, self.d_k)
            q_linears_dict[node_h_id] = q_mat
        for ids in range(data_size):
            node_t_id = int(edge_index.t()[ids][1])
            node_h_id = int(edge_index.t()[ids][0])
            k_m.append(k_linears_dict[node_t_id])
            v_m.append(v_linears_dict[node_t_id])
            q_m.append(q_linears_dict[node_h_id])
        k_m = torch.stack(k_m, dim=0)
        v_m = torch.stack(v_m, dim=0)
        q_m = torch.stack(q_m, dim=0)
        for relation in range(self.num_relations):
            idx = edge_type == int(relation)
            if idx.sum() == 0:
                continue
            k_node_vec = k_m[idx].view(-1, self.n_heads, self.d_k)
            v_node_vec = v_m[idx].view(-1, self.n_heads, self.d_k)
            q_node_vec = q_m[idx].view(-1, self.n_heads, self.d_k)
            k_node_vec = torch.bmm(k_node_vec.transpose(1, 0), self.relation_att[relation]).transpose(1, 0)
            res_att[idx] = (q_node_vec * k_node_vec).sum(dim=-1) * self.relation_pri[relation] / self.sqrt_dk
            res_msg[idx] = torch.bmm(v_node_vec.transpose(1, 0), self.relation_msg[relation]).transpose(1, 0)
        del k_linears_dict
        del v_linears_dict
        del q_linears_dict
        del k_m
        del v_m
        del q_m
        gc.collect()
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        gc.collect()
        end = time.time()
        print('Message aggregation time: {:.2f} s'.format(end - start))
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_maintype):
        start = time.time()
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for ids in range(aggr_out.size(0)):
            node_f = node_inp[ids]
            aggr_out_f = aggr_out[ids]
            target_types_id = int(node_maintype[ids])
            target_types = self.p.id2types[target_types_id]
            a_mat = []
            a_att = self.p.types2count[target_types]
            a_att = torch.softmax(torch.FloatTensor(a_att), dim=-1).to(node_inp.device)
            for stype in target_types.split(' '):
                stype_id = self.p.type2id[stype]
                w = self.a_linears[stype_id]
                bias = self.a_linears_bias[stype_id]
                mat = aggr_out_f @ w + bias
                a_mat.append(mat)
            a_mat = torch.stack(a_mat, 0)
            a_att = a_att.unsqueeze(-1)
            a_mat = a_att * a_mat
            a_mat = a_mat.sum(dim=0)
            trans_out = self.drop(a_mat)
            skip_linears = []
            skip_att = self.p.types2count[target_types]
            skip_att = torch.softmax(torch.FloatTensor(skip_att), dim=-1).to(node_inp.device)
            for skiptype in target_types.split(' '):
                skiptype_id = self.p.type2id[skiptype]
                skip_linear = self.skip[skiptype_id]
                skip_linears.append(skip_linear)
            skip_mat = torch.stack(skip_linears, dim=-1).to(node_inp.device)
            skip_mat = skip_att * skip_mat
            skip_mat = skip_mat.sum(dim=0)
            alpha = torch.sigmoid(skip_mat)
            if self.use_norm:
                no_linears = []
                no_att = self.p.types2count[target_types]
                no_att = torch.softmax(torch.FloatTensor(no_att), dim=-1).to(node_inp.device)
                for notype in target_types.split(' '):
                    notype_id = self.p.type2id[notype]
                    no_linear = self.norms[notype_id]
                    no_linears.append(no_linear)
                no_mat = []
                for no in no_linears:
                    no_mat.append(no(trans_out * alpha + node_f * (1 - alpha)))
                no_mat = torch.stack(no_mat, 0)
                no_att = no_att.unsqueeze(-1)
                no_mat = no_att * no_mat
                no_mat = no_mat.sum(dim=0)
                res[ids] = no_mat
            else:
                res[ids] = trans_out * alpha + node_f * (1 - alpha)
        end = time.time()
        print('Feature update time: {:.2f} s'.format(end - start))
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class DenseHGTConv(MessagePassing):

    def __init__(self, in_hid, out_hid, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(DenseHGTConv, self).__init__(node_dim=0, aggr='add', **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_mainmaintype
        self.num_relations = num_relations
        self.total_rel = num_mainmaintype * num_relations * num_mainmaintype
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.use_RTE = use_RTE
        self.att = None
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for t in range(num_mainmaintype):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.drop = nn.Dropout(dropout)
        if self.use_RTE:
            self.emb = RelTemporalEncoding(in_dim)
        glorot(self.relation_att)
        glorot(self.relation_msg)
        self.mid_linear = nn.Linear(out_dim, out_dim * 2)
        self.out_linear = nn.Linear(out_dim * 2, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, node_inp, node_type, edge_index, edge_type, edge_idssss):
        return self.propagate(edge_index, node_inp=node_inp, node_type=node_type, edge_type=edge_type, edge_idssss=edge_idssss)

    def message(self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_idssss):
        data_size = edge_index_i.size(0)
        res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)
        for source_type in range(self.num_maintype):
            sb = node_type_j == int(source_type)
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            for target_type in range(self.num_maintype):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_idssss[idx])
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(k_mat.transpose(1, 0), self.relation_att[relation_type]).transpose(1, 0)
                    res_att[idx] = (q_mat * k_mat).sum(dim=-1) * self.relation_pri[relation_type] / self.sqrt_dk
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(v_mat.transpose(1, 0), self.relation_msg[relation_type]).transpose(1, 0)
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_maintype):
            idx = node_type == int(target_type)
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx])) + node_inp[idx]
            if self.use_norm:
                trans_out = self.norms[target_type](trans_out)
            trans_out = self.drop(self.out_linear(F.gelu(self.mid_linear(trans_out)))) + trans_out
            res[idx] = self.out_norm(trans_out)
        return res

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_mainmaintype={}, num_mainmaintype={})'.format(self.__class__.__name__, self.in_dim, self.out_dim, self.num_maintype, self.num_relations)

class RelTemporalEncoding(nn.Module):

    def __init__(self, n_hid, max_len=240, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))

class Transe(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(Transe, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_ws = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, r, adj):
        rs = self.node_ws(node_inp)
        return rs

class Distmult(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(Distmult, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_ws = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, r, adj):
        rs = self.node_ws(node_inp)
        return rs

class Complex(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(Complex, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_ws = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, r, adj):
        rs = self.node_ws(node_inp)
        return rs

class Simple(MessagePassing):

    def __init__(self, in_dim, out_dim, num_maintype, num_rel, num_head, dropout, num_bases, params, use_norm, **kwargs):
        super(Simple, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.p = params
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_maintype = num_maintype
        self.num_relations = num_rel
        print('self.num_relations:', self.num_relations)
        self.total_rel = num_maintype * num_rel * 2 * num_maintype
        self.n_heads = num_head
        self.d_k = out_dim // num_head
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        self.node_ws = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, node_inp, node_maintype, type_count, edge_index, edge_type, r, adj):
        rs = self.node_ws(node_inp)
        return rs

class GeneralConv(nn.Module):

    def __init__(self, params, conv_name, in_hid, out_hid, num_maintype, num_rel, num_head, dropout, use_norm):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'mthgcl':
            self.base_conv = MTHGCLConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'dense_mthgcl':
            self.base_conv = DenseMTHGCLConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'hgcl':
            self.base_conv = HGCLConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'dense_hgcl':
            self.base_conv = DenseHGCLConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'gcn':
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == 'sagecn':
            self.base_conv = SAGEConv(in_hid, out_hid)
        elif self.conv_name == 'gat':
            self.base_conv = GATConv(in_hid, out_hid // num_head, heads=num_head)
        elif self.conv_name == 'rgcn':
            self.base_conv = RGCNConv(in_hid, out_hid, num_rel)
        elif self.conv_name == 'rgcnv2':
            self.base_conv = RGCNv2Conv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'rgcnv3':
            self.base_conv = RGCNv3Conv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'compgcn':
            self.base_conv = COMPConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'sage':
            self.base_conv = SAGE(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'norel_dense_mthgcl':
            self.base_conv = NoRelDenseMTHGCLConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'indrel_dense_mthgcl':
            self.base_conv = IndRelDenseMTHGCLConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'rel_dense_mthgcl':
            self.base_conv = RelDenseMTHGCLConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'rel_dense_mthgcl2':
            self.base_conv = RelDenseMTHGCLConv2(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'gcnv2':
            self.base_conv = GCN2Conv(out_hid, alpha=0.1, theta=0.5, layer=64)
        elif self.conv_name == 'gatv2':
            self.base_conv = GATv2Conv(in_hid, out_hid // num_head, heads=num_head)
        elif self.conv_name == 'gatv3':
            self.base_conv = GATv3Conv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'gatv4':
            self.base_conv = GATv4Conv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'transformer':
            self.base_conv = TransformerConv(in_hid, out_hid // num_head, heads=num_head)
        elif self.conv_name == 'fa':
            self.base_conv = FAConv(out_hid)
        elif self.conv_name == 'supergat':
            self.base_conv = SuperGATConv(in_hid, out_hid // num_head, heads=num_head)
        elif self.conv_name == 'heat':
            self.base_conv = HEATConv(in_hid, out_hid // num_head, num_maintype, num_rel, out_hid, out_hid, out_hid, heads=num_head)
        elif self.conv_name == 'eg':
            self.base_conv = EGConv(in_hid, out_hid, num_heads=num_head)
        elif self.conv_name == 'film':
            self.base_conv = FiLMConv(in_hid, out_hid, num_relations=num_rel)
        elif self.conv_name == 'hgt':
            self.base_conv = HGTConv(in_hid, out_hid, heads=num_head)
        elif self.conv_name == 'hrgat':
            self.base_conv = HRGATConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'lgtn':
            self.base_conv = LGTNConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'lgtnv2':
            self.base_conv = LGTNv2Conv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'lgtnv3':
            self.base_conv = LGTNv3Conv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'origin_sage':
            self.base_conv = OriginSAGE(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'transe':
            self.base_conv = Transe(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'distmult':
            self.base_conv = Distmult(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'complex':
            self.base_conv = Complex(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'simple':
            self.base_conv = Simple(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'gin':
            mlp = MLP([in_hid, in_hid, in_hid])
            self.base_conv = GINConv(nn=mlp)
        elif self.conv_name == 'fusedgat':
            self.base_conv = FusedGATConv(in_channels=in_hid, out_channels=out_hid, add_self_loops=False, edge_dim=None, concat=False)
        elif self.conv_name == 'ssg':
            self.base_conv = SSGConv(in_hid, out_hid, alpha=0.5)
        elif self.conv_name == 'rgat':
            self.base_conv = RGATConv(in_hid, in_hid, out_hid)
        elif self.conv_name == 'origin_compgcn':
            self.base_conv = Origin_COMPConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)
        elif self.conv_name == 'gtn':
            self.base_conv = GTNConv(in_hid, out_hid, num_maintype, num_rel, num_head, dropout, params.num_bases, params, use_norm)

    def forward(self, meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj, csr=None, csc=None, perm=None):
        if self.conv_name == 'mthgcl':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'dense_mthgcl':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'hgcl':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'dense_hgcl':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'gcn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'sagecn':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gat':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'rgcn':
            return self.base_conv(meta_xs, edge_index, edge_type)
        elif self.conv_name == 'rgcnv2':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'rgcnv3':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'compgcn':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'sage':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'norel_dense_mthgcl':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'indrel_dense_mthgcl':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'rel_dense_mthgcl':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'rel_dense_mthgcl2':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'gcnv2':
            return self.base_conv(meta_xs, meta_xs, edge_index)
        elif self.conv_name == 'gatv2':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'gatv3':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'gatv4':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'transformer':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'fa':
            return self.base_conv(meta_xs, meta_xs, edge_index)
        elif self.conv_name == 'supergat':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'heat':
            return self.base_conv(meta_xs, edge_index, node_maintype[:, 0], edge_type, r[edge_type])
        elif self.conv_name == 'eg':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'film':
            return self.base_conv(meta_xs, edge_index, edge_type)
        elif self.conv_name == 'hgt':
            return self.base_conv(meta_xs, edge_index, edge_type)
        elif self.conv_name == 'hrgat':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'lgtn':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'lgtnv2':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'lgtnv3':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'origin_sage':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'transe':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'distmult':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'complex':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'simple':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'gin':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'fusedgat':
            return self.base_conv(x=meta_xs, csr=csr, csc=csc, perm=perm)
        elif self.conv_name == 'ssg':
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == 'rgat':
            return self.base_conv(meta_xs, edge_index, edge_type)
        elif self.conv_name == 'origin_compgcn':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
        elif self.conv_name == 'gtn':
            return self.base_conv(meta_xs, node_maintype, type_count, edge_index, edge_type, r, adj)
