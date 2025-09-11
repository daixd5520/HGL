import os
from time import time
import datetime  # 导入 datetime 模块
import torch

from util import get_dataset, act, mkdir
from model.GNN_model import GNN, CurvatureParam
from model.GRACE_model import GRACE
# 若你在数据降维中使用过 SVDFeatureReduction，可保留；否则也可去掉导入
# from torch_geometric.transforms import SVDFeatureReduction


def pretrain(dataname, pretext, config, gpu, is_reduction=False):
    print(os.getcwd())
    path = os.path.join('./datasets', dataname)
    dataset = get_dataset(path, dataname)
    data = dataset[0]

    # if is_reduction:
    #     feature_reduce = SVDFeatureReduction(out_channels=100)
    #     data = feature_reduce(data)

    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # === 读取配置 ===
    input_dim = data.x.shape[1]
    output_dim = int(config['output_dim'])
    num_proj_dim = int(config.get('num_proj_dim', output_dim))
    activation = act(config['activation'])
    learning_rate = float(config['learning_rate'])
    weight_decay = float(config.get('weight_decay', 0.0))
    num_epochs = int(config.get('num_epochs', 1000))
    tau = float(config.get('tau', 0.5))
    gnn_type = config['gnn_type']
    num_layers = int(config['num_layers'])
    drop_edge_rate = float(config['drop_edge_rate'])
    drop_feature_rate = float(config['drop_feature_rate'])

    # === 双曲开关与可训练曲率 ===
    hyperbolic_backbone = bool(config.get('hyperbolic_backbone', True))
    init_c = float(config.get('curvature', 1.0))
    learnable_c = bool(config.get('learnable_curvature', True))
    min_c = float(config.get('min_curvature', 1e-4))
    max_c = float(config.get('max_curvature', 10.0))
    curv_param = CurvatureParam(init_c=init_c, min_c=min_c, max_c=max_c, learnable=learnable_c).to(device)

    # === 主干与预训练模型 ===
    gnn = GNN(input_dim, output_dim, activation, gnn_type, num_layers,
              hyperbolic=hyperbolic_backbone, curv=curv_param).to(device)

    if pretext == 'GRACE':
        pretrain_model = GRACE(gnn, output_dim, num_proj_dim,
                               drop_edge_rate, drop_feature_rate, tau,
                               hyperbolic=hyperbolic_backbone, curv=curv_param).to(device)
    else:
        raise NotImplementedError(f'Unsupported pretext: {pretext}')

    if hyperbolic_backbone:
        print(f'[Pretrain] Using Lorentz H^d with learnable curvature, init c={init_c}')

    # --- 修改部分：生成唯一且信息丰富的模型文件名 ---
    pre_trained_model_path = './pre_trained_gnn/'
    mkdir(pre_trained_model_path)
    
    # 1. 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 2. 构建包含双曲信息和时间戳的新文件名
    model_filename = "{}.{}.{}.hyp_{}.{}.{}.pth".format(
        dataname, 
        pretext, 
        gnn_type, 
        hyperbolic_backbone, 
        is_reduction, 
        timestamp
    )
    model_path = os.path.join(pre_trained_model_path, model_filename)
    # --- 修改结束 ---

    # === 优化器（给曲率更小 LR 更稳） ===
    base_lr = learning_rate
    params_main = [p for n, p in pretrain_model.named_parameters() if 'raw_c' not in n]
    params_c    = [pretrain_model.gnn.curv.raw_c]
    optimizer = torch.optim.Adam([
        {"params": params_main, "lr": base_lr, "weight_decay": weight_decay},
        {"params": params_c,    "lr": base_lr * 0.1, "weight_decay": 0.0},
    ])

    # === 训练 ===
    print("pre-training.")
    start = time()
    prev = start
    min_loss = 1e9

    for epoch in range(1, num_epochs + 1):
        pretrain_model.train()
        optimizer.zero_grad()
        loss = pretrain_model.compute_loss(data.x, data.edge_index)
        loss.backward()
        optimizer.step()

        now = time()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, c={curv_param.get().item():.6f}, this {now - prev:.2f}s, total {now - start:.2f}s')
        prev = now

        if min_loss > loss.item():
            min_loss = loss.item()
            torch.save(pretrain_model.gnn.state_dict(), model_path)
            # --- 修改部分：打印保存的完整文件名 ---
            print(f"+++ model saved ! {model_filename}")
            # --- 修改结束 ---

    print("=== Final ===")