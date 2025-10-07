import os
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from util import get_dataset, act
from model.GNN_model import GNN, CurvatureParam
import yaml
from yaml import SafeLoader


def load_config(config_path='/mnt/data1/Graph/HypGraphLoRA/config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config


def generate_embeddings(model_path, dataname='CiteSeer', output_path=None):
    """
    从预训练模型生成图数据集的节点embeddings

    Args:
        model_path: 预训练模型路径
        dataname: 数据集名称
        output_path: 输出embedding的路径，如果为None则自动生成
    """
    print(f"Loading model from: {model_path}")
    print(f"Dataset: {dataname}")

    # 加载配置
    config = load_config()
    dataset_config = config[dataname]

    # 确定是否为双曲模型（从路径判断）
    is_hyperbolic = 'hyp_True' in model_path or 'hyp=True' in model_path
    if 'nohyp' in model_path:
        is_hyperbolic = False

    print(f"Hyperbolic backbone: {is_hyperbolic}")

    # 加载数据集
    path = os.path.join('./datasets', dataname)
    dataset = get_dataset(path, dataname)
    data = dataset[0]

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    print(f"Using device: {device}")

    # 检查模型是否使用了特征降维（从模型文件名判断）
    is_reduction = 'True' in os.path.basename(model_path)

    if is_reduction:
        # 如果使用了降维，需要应用SVDFeatureReduction
        from torch_geometric.transforms import SVDFeatureReduction
        feature_reduce = SVDFeatureReduction(out_channels=100)
        data = feature_reduce(data)
        print("Applied feature reduction to 100 dimensions")

    # 从模型状态字典推断正确的维度配置
    state_dict = torch.load(model_path, map_location='cpu')

    # 推断层数和维度
    conv_layers = [k for k in state_dict.keys() if k.startswith('conv.') and '.lin_src.weight' in k]
    num_layers = len(conv_layers)

    # 获取每层的输出维度
    layer_dims = []
    for i in range(num_layers):
        weight_key = f'conv.{i}.lin_src.weight'
        if weight_key in state_dict:
            layer_dims.append(state_dict[weight_key].shape[0])
        else:
            # 如果找不到，使用配置中的默认值
            layer_dims.append(int(dataset_config['output_dim']))

    input_dim = data.x.shape[1]
    output_dim = layer_dims[-1]  # 最后一层的输出维度
    gnn_type = dataset_config['gnn_type']
    activation = act(dataset_config['activation'])

    print(f"Inferred model config: input_dim={input_dim}, layer_dims={layer_dims}, num_layers={num_layers}")

    # 双曲相关参数
    if is_hyperbolic:
        init_c = float(dataset_config.get('curvature', 1.0))
        learnable_c = bool(dataset_config.get('learnable_curvature', True))
        min_c = float(dataset_config.get('min_curvature', 1e-4))
        max_c = float(dataset_config.get('max_curvature', 10.0))
        curv_param = CurvatureParam(init_c=init_c, min_c=min_c, max_c=max_c, learnable=learnable_c).to(device)
    else:
        curv_param = None

    # 创建GNN模型 - 使用动态配置的维度
    gnn = GNN(input_dim=input_dim,
              out_dim=output_dim,  # 使用最后一层的输出维度
              activation=activation,
              gnn_type=gnn_type,
              gnn_layer_num=num_layers,
              hyperbolic=is_hyperbolic,
              curv=curv_param).to(device)

    # 如果模型有特殊的维度配置，需要手动调整
    # 对于这个特定的模型，我们知道它是：100 -> 512 -> 256
    if num_layers == 2 and len(layer_dims) == 2:
        print("Adjusting layer dimensions to match pre-trained model...")
        # 重新创建conv层以匹配预训练模型的维度
        gnn.conv = nn.ModuleList()

        dims = [input_dim] + layer_dims  # [100, 512, 256]
        for i in range(num_layers):
            in_c, out_c = dims[i], dims[i+1]
            if gnn_type == 'GCN':
                base = GCNConv(in_c, out_c, add_self_loops=True, normalize=True)
            elif gnn_type == 'Transformer':
                base = TransformerConv(in_c, out_c, heads=gnn.heads, dropout=0.0)
            else:
                base = GATConv(in_c, out_c, heads=gnn.heads, concat=False, dropout=0.0)

            if is_hyperbolic:
                from model.GNN_model import HyperbolicWrapper
                act_name = 'relu'  # 使用默认激活函数
                gnn.conv.append(HyperbolicWrapper(base, curv=curv_param, activation=act_name, dropout=0.0))
            else:
                gnn.conv.append(base)

    # 加载预训练权重
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        gnn.load_state_dict(state_dict)
        print("Successfully loaded model weights")
        # 确保模型在正确的设备上
        gnn = gnn.to(device)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 设置为评估模式
    gnn.eval()

    # 生成embeddings
    print("Generating embeddings...")
    with torch.no_grad():
        embeddings = gnn(data.x, data.edge_index)

        # 如果是双曲模型，需要转换到欧氏空间
        if is_hyperbolic:
            from util import lorentz_logmap0
            c = curv_param.get()
            embeddings = lorentz_logmap0(embeddings, c)

    # 转换为numpy数组
    embeddings_np = embeddings.cpu().numpy()
    print(f"Embeddings shape: {embeddings_np.shape}")

    # 保存embeddings
    if output_path is None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"./embeddings/{model_name}_embeddings.npy"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings_np)
    print(f"Embeddings saved to: {output_path}")

    # 同时保存标签（可选）
    labels_path = output_path.replace('_embeddings.npy', '_labels.npy')
    labels_np = data.y.cpu().numpy()
    np.save(labels_path, labels_np)
    print(f"Labels saved to: {labels_path}")

    return embeddings_np, labels_np


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate embeddings from pre-trained GNN model')
    parser.add_argument('--model_path', type=str,
                       default='/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/nohyp/CiteSeer.GRACE.GAT.True.pth',
                       help='Path to pre-trained model')
    parser.add_argument('--dataset', type=str, default='CiteSeer',
                       help='Dataset name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for embeddings (optional)')

    args = parser.parse_args()

    embeddings, labels = generate_embeddings(args.model_path, args.dataset, args.output)
