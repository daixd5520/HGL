import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from util import lorentz_logmap0
from model.GNN_model import CurvatureParam


def drop_feature(x, drop_prob: float):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x


class GRACE(nn.Module):
    """
    GRACE 支持双曲主干：
      - hyperbolic=False：直接用欧氏 gnn 输出 z (N,d)
      - hyperbolic=True ：gnn 输出 p∈H^d，先 logmap0 到切空间，再投影头 + InfoNCE
    """
    def __init__(self, gnn: nn.Module, feat_dim: int, proj_dim: int,
                 drop_edge_rate: float, drop_feature_rate: float, tau: float = 0.5,
                 hyperbolic: bool = False, curv: CurvatureParam | None = None):
        super().__init__()
        self.gnn = gnn
        self.tau = float(tau)
        self.drop_edge_rate = float(drop_edge_rate)
        self.drop_feature_rate = float(drop_feature_rate)
        self.hyperbolic = bool(hyperbolic)
        self.curv = curv

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.PReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def _encode(self, x, edge_index):
        z = self.gnn(x, edge_index)         # 欧氏:(N,d)  双曲:(N,1+d)
        if self.hyperbolic:
            c = self.curv.get()
            z = lorentz_logmap0(z, c)       # 切空间 (N,d)
        h = self.proj(z)
        h = F.normalize(h, p=2, dim=-1)
        return h

    def forward(self, x, edge_index):
        return self._encode(x, edge_index)

    @staticmethod
    def _sim(z1, z2):
        return torch.mm(z1, z2.t())

    def loss(self, z1, z2, mean: bool = True):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self._sim(z1, z1))
        between_sim = f(self._sim(z1, z2))

        def loss_fn(a, b):
            N = a.size(0)
            diag = torch.eye(N, device=a.device).bool()
            positives = b[diag]
            negatives = torch.cat([a[~diag].view(N, -1), b[~diag].view(N, -1)], dim=-1)
            denom = negatives.sum(dim=-1) + positives
            return -torch.log(positives / denom)

        l1 = loss_fn(refl_sim, between_sim)
        l2 = loss_fn(refl_sim.t(), between_sim.t())
        ret = 0.5 * (l1 + l2)
        return ret.mean() if mean else ret.sum()

    def compute_loss(self, x, edge_index):
        edge_index_1 = dropout_edge(edge_index, p=self.drop_edge_rate)[0]
        edge_index_2 = dropout_edge(edge_index, p=self.drop_edge_rate)[0]
        x_1 = drop_feature(x, self.drop_feature_rate)
        x_2 = drop_feature(x, self.drop_feature_rate)
        z1 = self.forward(x_1, edge_index_1)
        z2 = self.forward(x_2, edge_index_2)
        return self.loss(z1, z2)
