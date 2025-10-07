import torch
import os

def inspect_model(model_path):
    """检查预训练模型的状态字典结构"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    # 加载state_dict
    state_dict = torch.load(model_path, map_location='cpu')

    print("Model state_dict keys and shapes:")
    for key, tensor in state_dict.items():
        print(f"  {key}: {tensor.shape}")

    # 分析可能的配置
    if 'conv.0.lin_src.weight' in state_dict:
        weight = state_dict['conv.0.lin_src.weight']
        print("\nInferred configuration:")
        print(f"  Input dimension: {weight.shape[1]}")
        print(f"  Hidden dimension: {weight.shape[0]}")

    if 'conv.1.lin_src.weight' in state_dict:
        weight = state_dict['conv.1.lin_src.weight']
        print(f"  Output dimension: {weight.shape[0]}")

if __name__ == '__main__':
    model_path = '/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/nohyp/CiteSeer.GRACE.GAT.True.pth'
    inspect_model(model_path)
