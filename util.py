import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import to_dense_adj
import numpy as np
import yaml
from yaml import SafeLoader


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))


def act(act_type='leakyrelu'):
    if act_type == 'leakyrelu':
        return F.leaky_relu
    elif act_type == 'tanh':
        return torch.tanh
    elif act_type == 'relu':
        return F.relu
    elif act_type == 'prelu':
        return nn.PReLU()
    elif act_type == 'sigmiod':
        return F.sigmoid


def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'Computers', 'Photo']
    if (name == 'Computers') | (name == 'Photo'):
        return Amazon(path, name, T.NormalizeFeatures())
    else:
        return Planetoid(path, name, transform=T.NormalizeFeatures())


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class SMMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(SMMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target, ppr=None):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            if ppr is None:
                XX = torch.mean(kernels[:batch_size, :batch_size])
            else:
                XX = torch.mean(kernels[:batch_size, :batch_size] * ppr)
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


def get_ppr_matrix(dataset, alpha: float = 0.05):
    A_tilde = to_dense_adj(dataset.edge_index)[0]
    num_nodes = A_tilde.shape[0]
    D_tilde = torch.diag(1/torch.sqrt(A_tilde.sum(dim=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * torch.linalg.inv(torch.eye(num_nodes).to(A_tilde.device) - (1 - alpha) * H)


def get_ppr_weight(test_dataset):
    ppr_matrix = get_ppr_matrix(test_dataset)
    ppr_matrix[ppr_matrix == 0] = ppr_matrix[ppr_matrix != 0].min()
    ppr_matrix = torch.log(1 + 1 / ppr_matrix)
    ppr_weight = ppr_matrix / ppr_matrix.sum(1).unsqueeze(1) * ppr_matrix.shape[0]
    return ppr_weight


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")



def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def batched_gct_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, mask, tau = 0.5):
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / tau)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        idx = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(sim(z1[idx], z1))  # [B, N]
        between_sim = f(sim(z1[idx], z2))  # [B, N]

        losses.append(-torch.log(
            (mask[i * batch_size:(i + 1) * batch_size] * between_sim).sum(1)
            / (refl_sim.sum(1) + between_sim.sum(1)
               - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
    return torch.cat(losses)


def batched_smmd_loss(z1: torch.Tensor, z2, MMD, ppr_weight, batch_size):
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        ppr = ppr_weight[mask][:, mask]
        target = next(iter(z2))
        losses.append(MMD(z1[mask], target, ppr))

    return torch.stack(losses).mean()


def get_few_shot_mask(data, shot, dataname, device):
    np.random.seed(0)
    class_num = max(data.y) + 1
    y = data.y.cpu()
    selected = []
    if dataname in ['PubMed', 'CiteSeer', 'Cora']:
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        for i in range(class_num):
            selected.append(np.random.choice(torch.arange(len(y))[(y == i) & train_mask.cpu()], shot))
        train_mask = torch.zeros(len(y)).bool().to(device)
        train_mask[np.concatenate(selected)] = True
    else:
        for i in range(class_num):
            selected.append(np.random.choice(torch.arange(len(y))[y.cpu() == i], shot))
        train_mask = torch.zeros(len(y)).bool().to(device)
        val_mask = torch.zeros(len(y)).bool().to(device)
        test_mask = torch.zeros(len(y)).bool().to(device)
        train_mask[np.concatenate(selected)] = True
        index = np.arange(len(y))[~train_mask.cpu()]
        np.random.shuffle(index)
        val_mask[index[:int(len(index) * 0.2)]] = True
        test_mask[index[int(len(index) * 0.2):]] = True
    return train_mask, val_mask, test_mask


def get_parameter(args):
    config = yaml.load(open(args.para_config), Loader=SafeLoader)
    if args.few:
        if args.shot == 10:
            setting = '10shot'
        else:
            setting = '5shot'
    else:
        setting = 'public'
    args.wd1 = float(config[setting][args.test_dataset]['wd1'])
    args.wd2 = float(config[setting][args.test_dataset]['wd2'])
    args.wd3 = float(config[setting][args.test_dataset]['wd3'])
    args.lr1 = float(config[setting][args.test_dataset]['lr1'])
    args.lr2 = float(config[setting][args.test_dataset]['lr2'])
    args.lr3 = float(config[setting][args.test_dataset]['lr3'])
    args.l1 = float(config[setting][args.test_dataset]['l1'])
    args.l2 = float(config[setting][args.test_dataset]['l2'])
    args.l3 = float(config[setting][args.test_dataset]['l3'])
    args.l4 = float(config[setting][args.test_dataset]['l4'])
    args.num_epochs = config[setting][args.test_dataset]['num_epochs']
    return args

# ===== Hyperbolic (Lorentz model) ops =====
import math

def lorentz_expmap0(v: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    v: [N, d]  tangent at origin -> point on hyperboloid H^d
    Using a stable variant: t = sqrt(1 + c * ||v||^2), x = sqrt(c) * v
    """
    v2 = (v * v).sum(dim=-1, keepdim=True).clamp_min(eps)
    t = torch.sqrt(1.0 + c * v2)
    x = math.sqrt(c) * v
    return torch.cat([t, x], dim=-1)  # [N, 1+d]

def lorentz_logmap0(p: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    p: [N, 1+d] point on hyperboloid -> tangent at origin
    v = x / sqrt(c) * acosh(t) / sqrt(t^2 - 1)
    """
    t, x = p[:, :1], p[:, 1:]
    denom = (t * t - 1.0).clamp_min(eps).sqrt_()
    acosh = (t + (t * t - 1.0).clamp_min(eps).sqrt()).log()
    coef = acosh / denom
    v = (1.0 / math.sqrt(c)) * coef * x
    return v  # [N, d]

def lorentz_recompute_time(space: torch.Tensor, c: float = 1.0, eps: float = 1e-15) -> torch.Tensor:
    """
    给定空间分量 s: [N, d]，重建 time 分量 t = sqrt(1 + c * ||s||^2)
    返回 [N, 1+d]
    """
    t = torch.sqrt(1.0 + c * (space * space).sum(dim=-1, keepdim=True).clamp_min(eps))
    return torch.cat([t, space], dim=-1)

