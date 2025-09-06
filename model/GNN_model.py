import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.nn.dense.linear import Linear

# 你现有的 util.py 中已实现以下三个算子（Lorentz / 双曲）
from util import lorentz_expmap0, lorentz_logmap0, lorentz_recompute_time


# =========================
# 可训练曲率（全局共享）
# =========================
def _inverse_softplus(x: float) -> float:
    # softplus^{-1}(x) = log(exp(x) - 1)
    return math.log(math.expm1(x))

class CurvatureParam(nn.Module):
    """
    可训练曲率 c：c = clamp(softplus(raw_c), min_c, max_c)
    把本对象实例传给主干与 LoRA，即可共享同一曲率。
    """
    def __init__(self, init_c: float = 1.0, min_c: float = 1e-4, max_c: float = 10.0, learnable: bool = True):
        super().__init__()
        raw_init = _inverse_softplus(init_c)
        self.raw_c = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32), requires_grad=learnable)
        self.min_c = float(min_c)
        self.max_c = float(max_c)

    def get(self) -> Tensor:
        c = F.softplus(self.raw_c)
        c = torch.clamp(c, min=self.min_c, max=self.max_c)
        return c


# =========================
#  Hyperbolic LoRA (Lorentz)
# =========================
class HyperbolicLoRA(nn.Module):
    """
    在洛伦兹模型做 LoRA：
      x (切空间) --expmap0--> p -> A(空间分量) -> 重算time -> B -> 重算time --logmap0--> v
    返回切空间中的增量；由上层决定如何融合/再投到 H^d。
    """
    def __init__(self, in_features: int, out_features: int, r: int,
                 lora_alpha: float = 16.0, curv: Optional[CurvatureParam] = None):
        super().__init__()
        self.r = int(r)
        self.alpha = float(lora_alpha)
        self.curv = curv

        self.A = Linear(in_features, self.r, bias=False, weight_initializer='glorot')
        self.B = Linear(self.r, out_features, bias=False, weight_initializer='glorot')
        # LoRA 常规初始化：A ~ Kaiming, B ~ 0
        torch.nn.init.kaiming_normal_(self.A.weight)
        torch.nn.init.zeros_(self.B.weight)

    def forward(self, x: Tensor) -> Tensor:
        c = self.curv.get()
        p = lorentz_expmap0(x, c)         # [N, 1+d]
        s1 = self.A(p[:, 1:])
        p1 = lorentz_recompute_time(s1, c)
        s2 = self.B(p1[:, 1:])
        p2 = lorentz_recompute_time(s2, c)
        v  = lorentz_logmap0(p2, c)       # 切空间
        return (self.alpha / self.r) * v


# ===========================================
#  GATConv（线性投影里可注入 LoRA）
# ===========================================
class GATConv_lora(GATConv):
    """
    在 GAT 的源/目标线性投影处注入 LoRA：
      - hyperbolic=False: 欧氏 Sequential LoRA（A; B）
      - hyperbolic=True : 使用 HyperbolicLoRA（返回切空间增量）
    其它流程复用父类 GATConv 的实现。
    """
    def __init__(self, in_channels: int | Tuple[int, int], out_channels: int,
                 heads: int = 1, concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, add_self_loops: bool = True, edge_dim: int | None = None,
                 fill_value: float | Tensor | str = 'mean', bias: bool = True,
                 r: int = 32, hyperbolic: bool = False, lora_alpha: float = 16.0,
                 curv: Optional[CurvatureParam] = None, **kwargs):
        super().__init__(in_channels, out_channels, heads, concat, negative_slope, dropout,
                         add_self_loops, edge_dim, fill_value, bias, **kwargs)
        self.r = int(r)

        if isinstance(in_channels, int):
            if hyperbolic:
                self.lin_src = HyperbolicLoRA(in_channels, heads * out_channels, r, lora_alpha, curv)
                self.lin_dst = self.lin_src
            else:
                lin_a = Linear(in_channels, self.r, bias=False, weight_initializer='glorot')
                lin_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
                torch.nn.init.kaiming_normal_(lin_a.weight)
                torch.nn.init.zeros_(lin_b.weight)
                self.lin_src = nn.Sequential(lin_a, lin_b)
                self.lin_dst = self.lin_src
        else:
            if hyperbolic:
                self.lin_src = HyperbolicLoRA(in_channels[0], heads * out_channels, r, lora_alpha, curv)
                self.lin_dst = HyperbolicLoRA(in_channels[1], heads * out_channels, r, lora_alpha, curv)
            else:
                lin_src_a = Linear(in_channels[0], self.r, bias=False, weight_initializer='glorot')
                lin_src_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
                lin_dst_a = Linear(in_channels[1], self.r, bias=False, weight_initializer='glorot')
                lin_dst_b = Linear(self.r, heads * out_channels, bias=False, weight_initializer='glorot')
                for m in [lin_src_a, lin_dst_a]:
                    torch.nn.init.kaiming_normal_(m.weight)
                for m in [lin_src_b, lin_dst_b]:
                    torch.nn.init.zeros_(m.weight)
                self.lin_src = nn.Sequential(lin_src_a, lin_src_b)
                self.lin_dst = nn.Sequential(lin_dst_a, lin_dst_b)


# ======================================
#  任意 PyG 层的双曲包装：切空间→层→回 H^d
# ======================================
class HyperbolicWrapper(nn.Module):
    """
    让欧氏 PyG 卷积/注意力在 H^d_c 上工作：
      p_l --logmap0--> v  --conv--> v' --act/drop--> expmap0 --> p_{l+1}
    """
    def __init__(self, conv_module: nn.Module, curv: CurvatureParam,
                 activation: str | None = 'relu', dropout: float = 0.0):
        super().__init__()
        self.conv = conv_module
        self.curv = curv
        self.act = getattr(F, activation) if isinstance(activation, str) and activation not in (None, 'none', 'identity') else None
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, p: Tensor, edge_index: Tensor) -> Tensor:
        c = self.curv.get()
        v = lorentz_logmap0(p, c)
        v = self.conv(v, edge_index)
        if self.act is not None:
            v = self.act(v)
        v = self.drop(v)
        return lorentz_expmap0(v, c)


# =================
#  主干 (GNN)
# =================
def _act_name(a):
    return None if a in (None, 'none', 'identity') else a

class GNN(nn.Module):
    """
    - hyperbolic=False: 纯欧氏主干
    - hyperbolic=True : 每层切空间计算，再映回 H^d
    """
    def __init__(self, input_dim: int, out_dim: int, activation: str,
                 gnn_type: str = 'GAT', gnn_layer_num: int = 2,
                 hyperbolic: bool = False, curv: Optional[CurvatureParam] = None,
                 heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hyperbolic = bool(hyperbolic)
        self.curv = curv
        self.L = int(gnn_layer_num)
        self.gnn_type = gnn_type
        self.heads = int(heads)
        act_name = _act_name(activation)

        self.conv = nn.ModuleList()
        dims = [input_dim] + [out_dim] * self.L

        for i in range(self.L):
            in_c, out_c = dims[i], dims[i+1]
            if gnn_type == 'GCN':
                base = GCNConv(in_c, out_c, add_self_loops=True, normalize=True)
            elif gnn_type == 'Transformer':
                base = TransformerConv(in_c, out_c, heads=self.heads, dropout=dropout)
            else:
                base = GATConv(in_c, out_c, heads=self.heads, concat=False, dropout=dropout)

            if self.hyperbolic:
                self.conv.append(HyperbolicWrapper(base, curv=self.curv, activation=act_name, dropout=dropout))
            else:
                self.conv.append(base)

        self._eu_act = getattr(F, act_name) if (not self.hyperbolic and isinstance(act_name, str)) else None

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if not self.hyperbolic:
            h = x
            for i in range(self.L - 1):
                h = self.conv[i](h, edge_index)
                if self._eu_act is not None:
                    h = self._eu_act(h)
            h = self.conv[-1](h, edge_index)
            return h  # 欧氏 (N, d)
        else:
            c = self.curv.get()
            p = lorentz_expmap0(x, c)
            for i in range(self.L):
                p = self.conv[i](p, edge_index)
            return p      # 超曲 (N, 1+d)


# =================
#  并联 LoRA
# =================
class GNNLoRA(nn.Module):
    """
    并联 LoRA：冻结主干 gnn（可欧氏/可双曲）。
    若 hyperbolic=True：
      - LoRA 分支各层同样在 H^d（层内切空间计算/层间回 H^d）
      - 融合在 H^d：相加空间分量 + 重算 time
      - 输出默认回切空间（便于下游损失）
    """
    def __init__(self, input_dim: int, out_dim: int, activation: str, gnn: GNN,
                 gnn_type: str = 'GAT', gnn_layer_num: int = 2,
                 r: int = 32, hyperbolic: bool = False, lora_alpha: float = 16.0,
                 curv: Optional[CurvatureParam] = None,
                 heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gnn = gnn
        self.curv = curv
        self.hyperbolic = bool(hyperbolic)
        self.L = int(gnn_layer_num)
        self.gnn_type = gnn_type
        self.heads = int(heads)
        act_name = _act_name(activation)

        # LoRA 分支
        self.conv = nn.ModuleList()
        dims = [input_dim] + [out_dim] * self.L
        for i in range(self.L):
            in_c, out_c = dims[i], dims[i+1]
            if gnn_type == 'GCN':
                base = GCNConv(in_c, out_c, add_self_loops=True, normalize=True)
            elif gnn_type == 'Transformer':
                base = TransformerConv(in_c, out_c, heads=self.heads, dropout=dropout)
            else:
                # 在 GAT 的投影里注入 LoRA（欧氏/双曲包装由 HyperbolicWrapper 负责）
                base = GATConv_lora(in_c, out_c, heads=self.heads, concat=False, dropout=dropout,
                                    r=r, hyperbolic=False,  # 投影处自身是欧氏序列；几何外包由 Wrapper 处理
                                    lora_alpha=lora_alpha, curv=self.curv)
            if self.hyperbolic:
                self.conv.append(HyperbolicWrapper(base, curv=self.curv, activation=act_name, dropout=dropout))
            else:
                self.conv.append(base)

    def forward(self, x: Tensor, edge_index: Tensor, return_euclid: bool = True):
        # 主干输出
        main_out = self.gnn(x, edge_index)

        if not self.hyperbolic:
            # 欧氏并联（与你原始实现一致）
            h = x
            for i in range(self.L - 1):
                h = self.conv[i](h, edge_index)
            lora_out = self.conv[-1](h, edge_index)
            fused = main_out + lora_out
            return fused, main_out, lora_out

        # 超曲并联
        c = self.curv.get()
        p_lora = lorentz_expmap0(x, c)
        for i in range(self.L):
            p_lora = self.conv[i](p_lora, edge_index)

        p_main = main_out                 # (N, 1+d)
        s_fused = p_main[:, 1:] + p_lora[:, 1:]
        p_fused = lorentz_recompute_time(s_fused, c)

        if return_euclid:
            v_fused = lorentz_logmap0(p_fused, c)
            v_main  = lorentz_logmap0(p_main,  c)
            v_lora  = lorentz_logmap0(p_lora,  c)
            return v_fused, v_main, v_lora
        else:
            return p_fused, p_main, p_lora
