from torch_geometric.nn import GCNConv, GATConv, TransformerConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from typing import Tuple
from torch import Tensor


class GNN(torch.nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn_type='TransformerConv', gnn_layer_num=2):
        super().__init__()
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation
        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if gnn_layer_num < 1:
            raise ValueError('GNN layer_num should >=1 but you set {}'.format(gnn_layer_num))
        elif gnn_layer_num == 1:
            self.conv = nn.ModuleList([GraphConv(input_dim, out_dim)])
        elif gnn_layer_num == 2:
            self.conv = nn.ModuleList([GraphConv(input_dim, 2 * out_dim), GraphConv(2 * out_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, 2 * out_dim)]
            for i in range(gnn_layer_num - 2):
                layers.append(GraphConv(2 * out_dim, 2 * out_dim))
            layers.append(GraphConv(2 * out_dim, out_dim))
            self.conv = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for conv in self.conv[0:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
        node_emb = self.conv[-1](x, edge_index)
        return node_emb


class GATConv_lora(GATConv):
    def __init__(self, in_channels: int | Tuple[int, int], out_channels: int, heads: int = 1, concat: bool = True, negative_slope: float = 0.2, dropout: float = 0, add_self_loops: bool = True, edge_dim: int | None = None, fill_value: float | Tensor | str = 'mean', bias: bool = True, r: int = 32, **kwargs):
        super().__init__(in_channels, out_channels, heads, concat, negative_slope, dropout, add_self_loops, edge_dim, fill_value, bias, **kwargs)
        self.r = r
        
        if isinstance(in_channels, int):
            self.lin_src_a = Linear(in_channels, self.r, bias=False, weight_initializer='glorot')
            self.lin_src_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_src = nn.Sequential(self.lin_src_a, self.lin_src_b)
            self.lin_dst = self.lin_src
        else:
            self.lin_src_a = Linear(in_channels[0], self.r, bias=False, weight_initializer='glorot')
            self.lin_src_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_src = nn.Sequential(self.lin_src_a, self.lin_src_b)
            self.lin_dst_a = Linear(in_channels[1], self.r, bias=False, weight_initializer='glorot')
            self.lin_dst_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_dst = nn.Sequential(self.lin_dst_a, self.lin_dst_b)

        self.reset_parameters_lora()

    def reset_parameters_lora(self):
        torch.nn.init.kaiming_normal_(self.lin_src[0].weight)
        torch.nn.init.zeros_(self.lin_src[1].weight)
        torch.nn.init.kaiming_normal_(self.lin_dst[0].weight)
        torch.nn.init.zeros_(self.lin_dst[1].weight)
    

class GNNLoRA(torch.nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn, gnn_type='GAT', gnn_layer_num=2, r=32):
        super().__init__()
        self.gnn = gnn
        self.gnn_layer_num = gnn_layer_num
        self.activation = activation
        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv_lora
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        if gnn_layer_num < 1:
            raise ValueError('GNN layer_num should >=1 but you set {}'.format(gnn_layer_num))
        elif gnn_layer_num == 1:
            self.conv = nn.ModuleList([GraphConv(input_dim, out_dim, r=r)])
        elif gnn_layer_num == 2:
            self.conv = nn.ModuleList([GraphConv(input_dim, 2 * out_dim, r=r), GraphConv(2 * out_dim, out_dim, r=r)])
        else:
            layers = [GraphConv(input_dim, 2 * out_dim, r=r)]
            for i in range(gnn_layer_num - 2):
                layers.append(GraphConv(2 * out_dim, 2 * out_dim, r=r))
            layers.append(GraphConv(2 * out_dim, out_dim, r=r))
            self.conv = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for i in range(self.gnn_layer_num - 1):
            conv1 = self.gnn.conv[i]
            conv2 = self.conv[i]
            x = conv1(x, edge_index) + conv2(x, edge_index)
        node_emb1 = self.gnn.conv[-1](x, edge_index)
        node_emb2 = self.conv[-1](x, edge_index)
        return node_emb1 + node_emb2, node_emb1, node_emb2