import torch
import os

def inspect_model(model_path):
    """检查预训练模型的状态字典结构"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    try:
        # 加载state_dict
        state_dict = torch.load(model_path, map_location='cpu')

        print(f"Model: {os.path.basename(model_path)}")
        print("State_dict keys and shapes:")
        for key, tensor in state_dict.items():
            print(f"  {key}: {tensor.shape}")

        # 分析可能的配置
        if 'conv.0.conv.lin_src.weight' in state_dict:
            weight = state_dict['conv.0.conv.lin_src.weight']
            print("\nConv layer config:")
            print(f"  Input dimension: {weight.shape[1]}")
            print(f"  Hidden dimension: {weight.shape[0]}")

        elif 'conv.0.lin_src.weight' in state_dict:
            weight = state_dict['conv.0.lin_src.weight']
            print("\nGNN layer config:")
            print(f"  Input dimension: {weight.shape[1]}")
            print(f"  Hidden dimension: {weight.shape[0]}")

        if 'curv.raw_c' in state_dict:
            print(f"  Curvature parameter: {state_dict['curv.raw_c']}")

        print("-" * 50)

    except Exception as e:
        print(f"Error loading {model_path}: {e}")

if __name__ == '__main__':
    # 检查几个失败的模型
    failed_models = [
        '/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/CiteSeer.GRACE.GAT.curv_0p1000.hyp_True.False.pth',
        '/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/CiteSeer.GRACE.GAT.hyp_True.True.20250912-232342.pth'
    ]

    for model in failed_models:
        inspect_model(model)
