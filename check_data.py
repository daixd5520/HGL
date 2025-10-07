import os
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import SVDFeatureReduction, NormalizeFeatures
import torch_geometric.transforms as T

# 加载原始CiteSeer数据（不使用NormalizeFeatures）
path = os.path.join('./datasets', 'CiteSeer')
dataset_raw = Planetoid(path, 'CiteSeer')  # 不使用变换
data_raw = dataset_raw[0]
print(f"Raw data shape: {data_raw.x.shape}")

# 使用NormalizeFeatures
dataset_norm = Planetoid(path, 'CiteSeer', transform=T.NormalizeFeatures())
data_norm = dataset_norm[0]
print(f"Normalized data shape: {data_norm.x.shape}")

# 应用降维到原始数据
feature_reduce = SVDFeatureReduction(out_channels=100)
data_reduced = feature_reduce(data_raw)
print(f"Reduced raw data shape: {data_reduced.x.shape}")

# 检查数据值
print(f"Raw data sum: {data_raw.x.sum().item():.6f}")
print(f"Normalized data sum: {data_norm.x.sum().item():.6f}")
print(f"Raw data first 10 values: {data_raw.x[0, :10]}")
print(f"Normalized data first 10 values: {data_norm.x[0, :10]}")
