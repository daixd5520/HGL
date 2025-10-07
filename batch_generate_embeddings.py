import os
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from util import get_dataset, act
from model.GNN_model import GNN, CurvatureParam
import yaml
from yaml import SafeLoader
import glob


def load_config(config_path='/mnt/data1/Graph/HypGraphLoRA/config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config


def get_all_citeseer_models(model_dir='/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn'):
    """获取所有CiteSeer相关的模型文件"""
    model_files = []

    # 主目录下的CiteSeer模型
    main_dir = model_dir
    citeseer_pattern = os.path.join(main_dir, 'CiteSeer.*.pth')
    model_files.extend(glob.glob(citeseer_pattern))

    # nohyp目录下的CiteSeer模型
    nohyp_dir = os.path.join(model_dir, 'nohyp')
    nohyp_citeseer_pattern = os.path.join(nohyp_dir, 'CiteSeer.*.pth')
    model_files.extend(glob.glob(nohyp_citeseer_pattern))

    return sorted(model_files)


def create_model_from_checkpoint(model_path, dataset_config, input_dim, device):
    """从checkpoint创建匹配的模型"""
    state_dict = torch.load(model_path, map_location='cpu')

    # 检查模型结构类型
    has_hyperbolic_wrapper = any('conv.0.conv.' in key for key in state_dict.keys())
    is_hyperbolic = 'hyp_True' in model_path or 'hyp=True' in model_path
    if 'nohyp' in model_path:
        is_hyperbolic = False

    print(f"  Model type: {'Hyperbolic' if is_hyperbolic else 'Euclidean'}, Wrapper: {has_hyperbolic_wrapper}")

    if has_hyperbolic_wrapper:
        # 处理HyperbolicWrapper包装的模型
        return create_hyperbolic_wrapper_model(model_path, dataset_config, input_dim, device)
    else:
        # 处理普通GNN模型（可能有降维）
        return create_standard_gnn_model(model_path, dataset_config, input_dim, device)


def create_hyperbolic_wrapper_model(model_path, dataset_config, input_dim, device):
    """创建HyperbolicWrapper包装的模型"""
    state_dict = torch.load(model_path, map_location='cpu')

    # 推断层数
    conv_layers = [k for k in state_dict.keys() if k.startswith('conv.') and 'conv.att_src' in k]
    num_layers = len(conv_layers)

    print(f"  Layers: {num_layers}, Input dim: {input_dim}")

    # 创建每一层的HyperbolicWrapper
    conv_modules = nn.ModuleList()

    for i in range(num_layers):
        # 创建基础GATConv
        if i == 0:
            in_dim = input_dim
        else:
            # 从权重推断输入维度
            weight_key = f'conv.{i}.conv.lin_src.weight'
            in_dim = state_dict[weight_key].shape[1]

        # 从权重推断输出维度和heads
        weight_key = f'conv.{i}.conv.lin_src.weight'
        weight_shape = state_dict[weight_key].shape  # [out_channels, in_channels]

        # 检查attention权重来推断heads
        att_key = f'conv.{i}.conv.att_src'
        att_shape = state_dict[att_key].shape  # [heads, 1, out_channels] for concat=False

        heads = att_shape[0]
        out_dim_per_head = att_shape[2]

        # 验证维度一致性
        expected_out_channels = heads * out_dim_per_head if heads > 1 else out_dim_per_head
        assert weight_shape[0] == expected_out_channels, f"Dimension mismatch: weight {weight_shape[0]}, expected {expected_out_channels}"

        print(f"    Layer {i}: in_dim={in_dim}, out_dim={out_dim_per_head}, heads={heads}")

        base = GATConv(in_dim, out_dim_per_head, heads=heads, concat=False, dropout=0.0)

        # 创建对应的curvature参数
        curv_key = f'conv.{i}.curv.raw_c'
        if curv_key in state_dict:
            raw_c_value = state_dict[curv_key].item()
            curv_param = CurvatureParam(init_c=float('nan'), min_c=1e-4, max_c=10.0, learnable=False)
            curv_param.raw_c.data = torch.tensor(raw_c_value)
        else:
            curv_param = CurvatureParam(init_c=1.0, min_c=1e-4, max_c=10.0, learnable=False)

        # 创建HyperbolicWrapper
        from model.GNN_model import HyperbolicWrapper
        wrapper = HyperbolicWrapper(base, curv=curv_param, activation='relu', dropout=0.0)
        conv_modules.append(wrapper)

    # 创建一个简单的容器来保存这些层
    class HyperbolicGNN(nn.Module):
        def __init__(self, conv_modules, curv_param):
            super().__init__()
            self.conv = conv_modules
            self.curv_param = curv_param

        def forward(self, x, edge_index):
            # 首先将欧氏输入转换为洛伦兹空间
            from util import lorentz_expmap0
            c = self.curv_param.get()
            p = lorentz_expmap0(x, c)

            # 在洛伦兹空间进行计算
            for conv in self.conv:
                p = conv(p, edge_index)

            return p

    model = HyperbolicGNN(conv_modules, curv_param)

    # 加载权重
    # 需要手动加载每个子模块的权重
    for i, conv in enumerate(model.conv):
        prefix = f'conv.{i}.'
        conv_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        conv.load_state_dict(conv_state)

    model = model.to(device)
    model.eval()
    return model, True


def create_standard_gnn_model(model_path, dataset_config, input_dim, device):
    """创建标准的GNN模型（可能有降维）"""
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

    print(f"  Layers: {num_layers}, Layer dims: {layer_dims}, Input dim: {input_dim}")

    # 确定是否为双曲模型
    is_hyperbolic = 'hyp_True' in model_path or 'hyp=True' in model_path
    if 'nohyp' in model_path:
        is_hyperbolic = False

    # 创建GNN模型
    gnn = GNN(input_dim=input_dim,
              out_dim=layer_dims[-1] if layer_dims else int(dataset_config['output_dim']),
              activation=act(dataset_config['activation']),
              gnn_type=dataset_config['gnn_type'],
              gnn_layer_num=num_layers,
              hyperbolic=is_hyperbolic,
              curv=None).to(device)

    # 如果模型有特殊的维度配置，需要手动调整
    if num_layers == 2 and len(layer_dims) == 2:
        gnn.conv = nn.ModuleList()
        dims = [input_dim] + layer_dims
        for i in range(num_layers):
            in_c, out_c = dims[i], dims[i+1]
            if dataset_config['gnn_type'] == 'GCN':
                base = GCNConv(in_c, out_c, add_self_loops=True, normalize=True)
            elif dataset_config['gnn_type'] == 'Transformer':
                base = TransformerConv(in_c, out_c, heads=gnn.heads, dropout=0.0)
            else:
                base = GATConv(in_c, out_c, heads=gnn.heads, concat=False, dropout=0.0)

            if is_hyperbolic:
                from model.GNN_model import HyperbolicWrapper
                act_name = 'relu'
                curv_param = CurvatureParam(init_c=1.0, min_c=1e-4, max_c=10.0, learnable=False).to(device)
                gnn.conv.append(HyperbolicWrapper(base, curv=curv_param, activation=act_name, dropout=0.0))
            else:
                gnn.conv.append(base)

    # 加载权重
    gnn.load_state_dict(state_dict)
    gnn = gnn.to(device)
    gnn.eval()

    return gnn, is_hyperbolic


def generate_embeddings_for_model(model_path, dataset_config, device, original_data):
    """为单个模型生成embeddings"""
    print(f"Processing model: {os.path.basename(model_path)}")

    try:
        # 检查模型是否需要降维
        state_dict = torch.load(model_path, map_location='cpu')
        has_hyperbolic_wrapper = any('conv.0.conv.' in key for key in state_dict.keys())

        if has_hyperbolic_wrapper:
            # HyperbolicWrapper模型使用原始输入维度
            data = original_data
            input_dim = data.x.shape[1]
        else:
            # 检查是否需要降维
            if 'conv.0.lin_src.weight' in state_dict:
                expected_input_dim = state_dict['conv.0.lin_src.weight'].shape[1]
                if expected_input_dim == 100 and original_data.x.shape[1] != 100:
                    # 需要降维
                    from torch_geometric.transforms import SVDFeatureReduction
                    feature_reduce = SVDFeatureReduction(out_channels=100)
                    data = feature_reduce(original_data.clone())
                    print("    Applied feature reduction to 100 dimensions")
                    input_dim = 100
                else:
                    data = original_data
                    input_dim = data.x.shape[1]
            else:
                data = original_data
                input_dim = data.x.shape[1]

        gnn, is_hyperbolic = create_model_from_checkpoint(model_path, dataset_config, input_dim, device)

        # 生成embeddings
        with torch.no_grad():
            print(f"    Input data shape: {data.x.shape}, dtype: {data.x.dtype}")
            embeddings = gnn(data.x, data.edge_index)
            print(f"    Output embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

            # 如果是双曲模型，需要转换到欧氏空间
            if is_hyperbolic:
                from util import lorentz_logmap0
                curv_param = CurvatureParam(init_c=1.0, min_c=1e-4, max_c=10.0, learnable=False).to(device)
                embeddings = lorentz_logmap0(embeddings, curv_param.get())

        embeddings_np = embeddings.cpu().numpy()
        print(f"  Generated embeddings with shape: {embeddings_np.shape}")
        return embeddings_np

    except Exception as e:
        print(f"  Error processing {os.path.basename(model_path)}: {e}")
        return None


def batch_generate_embeddings(model_dir='/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn',
                             output_dir='/mnt/data1/Graph/HypGraphLoRA/embeddings',
                             dataset_name='CiteSeer'):
    """批量生成所有CiteSeer模型的embeddings"""
    print(f"Starting batch generation for {dataset_name} models...")

    # 获取所有CiteSeer模型
    model_files = get_all_citeseer_models(model_dir)
    print(f"Found {len(model_files)} CiteSeer models:")
    for model in model_files:
        print(f"  {os.path.basename(model)}")

    if not model_files:
        print("No CiteSeer models found!")
        return

    # 加载配置
    config = load_config()
    dataset_config = config[dataset_name]

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据集
    path = os.path.join('./datasets', dataset_name)
    dataset = get_dataset(path, dataset_name)
    data = dataset[0]
    print(f"Loaded data shape: {data.x.shape}")
    data = data.to(device)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 为每个模型生成embeddings
    results = {}
    for model_path in model_files:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        embeddings = generate_embeddings_for_model(model_path, dataset_config, device, data)

        if embeddings is not None:
            # 保存embeddings
            emb_path = os.path.join(output_dir, f"{model_name}_embeddings.npy")
            np.save(emb_path, embeddings)
            results[model_name] = emb_path
            print(f"  Saved to: {emb_path}")

    print(f"\nBatch generation completed! Processed {len(results)} models successfully.")
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Batch generate embeddings for all CiteSeer models')
    parser.add_argument('--model_dir', type=str,
                       default='/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn',
                       help='Directory containing pre-trained models')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/data1/Graph/HypGraphLoRA/embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--dataset', type=str, default='CiteSeer',
                       help='Dataset name')

    args = parser.parse_args()

    results = batch_generate_embeddings(args.model_dir, args.output_dir, args.dataset)

    # 打印结果摘要
    print("\nGenerated embeddings for:")
    for model_name, path in results.items():
        print(f"  {model_name}: {path}")
