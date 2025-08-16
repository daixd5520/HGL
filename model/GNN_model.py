from torch_geometric.nn import GCNConv, GATConv, TransformerConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from typing import Tuple
from torch import Tensor

from util import lorentz_expmap0, lorentz_logmap0, lorentz_recompute_time

class HyperbolicLoRA(nn.Module):
    """
    在洛伦兹空间做 LoRA：x -> expmap0 -> A(·) -> 重算time -> B(·) -> 重算time -> logmap0
    A: Linear(d_in -> r), B: Linear(r -> d_out);  B 初始为 0, A 用 Kaiming
    """
    def __init__(self, in_features, out_features, r, lora_alpha=16.0, curvature=1.0):
        super().__init__()
        from torch_geometric.nn.dense.linear import Linear
        self.A = Linear(in_features, r, bias=False, weight_initializer='glorot')
        self.B = Linear(r, out_features, bias=False, weight_initializer='glorot')
        torch.nn.init.kaiming_normal_(self.A.weight)
        torch.nn.init.zeros_(self.B.weight)
        self.alpha = lora_alpha
        self.r = r
        self.c = curvature

    def forward(self, x):
        # x: [N, d_in]  （作为原点切空间向量）
        p0 = lorentz_expmap0(x, c=self.c)             # -> [N, 1+d_in]
        s1 = self.A(p0[:, 1:])                        # A 只作用于空间分量
        p1 = lorentz_recompute_time(s1, c=self.c)     # 重算 time
        s2 = self.B(p1[:, 1:])                        # B 只作用于空间分量
        p2 = lorentz_recompute_time(s2, c=self.c)
        v  = lorentz_logmap0(p2, c=self.c)            # 回到切空间
        return (self.alpha / self.r) * v

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
    def __init__(self, in_channels: int | Tuple[int, int], out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0, add_self_loops: bool = True,
                 edge_dim: int | None = None, fill_value: float | Tensor | str = 'mean', bias: bool = True,
                 r: int = 32, hyperbolic: bool = False, lora_alpha: float = 16.0, curvature: float = 1.0, **kwargs):
        super().__init__(in_channels, out_channels, heads, concat, negative_slope, dropout,
                         add_self_loops, edge_dim, fill_value, bias, **kwargs)
        self.r = r

        if isinstance(in_channels, int):
            if hyperbolic:
                self.lin_src = HyperbolicLoRA(in_channels, heads * out_channels, r, lora_alpha, curvature)
                self.lin_dst = self.lin_src
            else:
                self.lin_src_a = Linear(in_channels, self.r, bias=False, weight_initializer='glorot')
                self.lin_src_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
                self.lin_src = nn.Sequential(self.lin_src_a, self.lin_src_b)
                self.lin_dst = self.lin_src
        else:
            if hyperbolic:
                self.lin_src = HyperbolicLoRA(in_channels[0], heads * out_channels, r, lora_alpha, curvature)
                self.lin_dst = HyperbolicLoRA(in_channels[1], heads * out_channels, r, lora_alpha, curvature)
            else:
                self.lin_src_a = Linear(in_channels[0], self.r, bias=False, weight_initializer='glorot')
                self.lin_src_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
                self.lin_src = nn.Sequential(self.lin_src_a, self.lin_src_b)
                self.lin_dst_a = Linear(in_channels[1], self.r, bias=False, weight_initializer='glorot')
                self.lin_dst_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
                self.lin_dst = nn.Sequential(self.lin_dst_a, self.lin_dst_b)

        # 仍保持原有的 reset 对欧氏 LoRA 生效；双曲 LoRA 本身已按 LoRA 规则初始化
        if not isinstance(self.lin_src, HyperbolicLoRA):
            self.reset_parameters_lora()

    def reset_parameters_lora(self):
        torch.nn.init.kaiming_normal_(self.lin_src[0].weight)
        torch.nn.init.zeros_(self.lin_src[1].weight)
        torch.nn.init.kaiming_normal_(self.lin_dst[0].weight)
        torch.nn.init.zeros_(self.lin_dst[1].weight)
    

class GNNLoRA(torch.nn.Module):
    def __init__(self, input_dim, out_dim, activation, gnn, gnn_type='GAT', gnn_layer_num=2,
                 r=32, hyperbolic=False, lora_alpha=16.0, curvature=1.0):
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
            raise ValueError(f'GNN layer_num should >=1 but you set {gnn_layer_num}')
        elif gnn_layer_num == 1:
            self.conv = nn.ModuleList([GraphConv(input_dim, out_dim, r=r,
                                                 hyperbolic=hyperbolic, lora_alpha=lora_alpha, curvature=curvature)])
        elif gnn_layer_num == 2:
            self.conv = nn.ModuleList([
                GraphConv(input_dim, 2 * out_dim, r=r, hyperbolic=hyperbolic, lora_alpha=lora_alpha, curvature=curvature),
                GraphConv(2 * out_dim, out_dim, r=r, hyperbolic=hyperbolic, lora_alpha=lora_alpha, curvature=curvature)
            ])
        else:
            layers = [GraphConv(input_dim, 2 * out_dim, r=r, hyperbolic=hyperbolic, lora_alpha=lora_alpha, curvature=curvature)]
            for _ in range(gnn_layer_num - 2):
                layers.append(GraphConv(2 * out_dim, 2 * out_dim, r=r, hyperbolic=hyperbolic, lora_alpha=lora_alpha, curvature=curvature))
            layers.append(GraphConv(2 * out_dim, out_dim, r=r, hyperbolic=hyperbolic, lora_alpha=lora_alpha, curvature=curvature))
            self.conv = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for i in range(self.gnn_layer_num - 1):
            conv1 = self.gnn.conv[i]
            conv2 = self.conv[i]
            x = conv1(x, edge_index) + conv2(x, edge_index)
        node_emb1 = self.gnn.conv[-1](x, edge_index)
        node_emb2 = self.conv[-1](x, edge_index)
        return node_emb1 + node_emb2, node_emb1, node_emb2