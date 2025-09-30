import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj, add_remaining_self_loops
from torch_geometric.loader import DataLoader

from util import get_dataset, act, SMMDLoss, mkdir, get_ppr_weight
from util import get_few_shot_mask, batched_smmd_loss, batched_gct_loss
from util import get_adaptive_loss_weights, augment_few_shot_features, get_balanced_few_shot_mask
from model.GNN_model import GNN, GNNLoRA, HyperbolicLoRA, CurvatureParam

import datetime

class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.initialize()

    def forward(self, x): return self.fc(x)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x): return self.fc(x)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration in few-shot learning"""
    def __init__(self, initial_temp=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temp, dtype=torch.float32))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.01)  # Prevent division by zero


def get_few_shot_scheduler(optimizer, num_epochs):
    """Progressive learning rate scheduler for few-shot learning"""
    warmup_epochs = min(20, num_epochs // 4)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[warmup_epochs]
    )


def transfer(args, config, gpu_id, is_reduction):
    # 当仅有一张可见卡时，统一回落至 cuda:0；否则按传入 gpu_id 选择
    if torch.cuda.is_available():
        num_visible = torch.cuda.device_count()
        if num_visible >= 1:
            selected = 0 if num_visible == 1 else int(gpu_id)
            device = torch.device(f'cuda:{selected}')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    # ---- load data ----
    pretrian_datapath = os.path.join('./datasets', args.pretrain_dataset)
    test_datapath = os.path.join('./datasets', args.test_dataset)
    pretrain_dataset = get_dataset(pretrian_datapath, args.pretrain_dataset)[0]
    test_dataset = get_dataset(test_datapath, args.test_dataset)[0]

    pretrain_dataset.edge_index = add_remaining_self_loops(pretrain_dataset.edge_index)[0]
    test_dataset.edge_index = add_remaining_self_loops(test_dataset.edge_index)[0]
    pretrain_dataset = pretrain_dataset.to(device)
    test_dataset = test_dataset.to(device)

    # ---- adjacency and weights ----
    target_adj = to_dense_adj(test_dataset.edge_index)[0]
    pos_weight = float(test_dataset.x.shape[0] * test_dataset.x.shape[0] - test_dataset.edge_index.shape[1]) / max(1, test_dataset.edge_index.shape[1])
    weight_mask = target_adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0), device=device)
    weight_tensor[weight_mask] = pos_weight / 10.0

    # ---- curvature (shared & learnable) ----
    init_c = float(args.curvature)
    learnable_c = True
    min_c = float(getattr(args, 'min_curvature', 1e-4))
    max_c = float(getattr(args, 'max_curvature', 10.0))
    curv_param = CurvatureParam(init_c=init_c, min_c=min_c, max_c=max_c, learnable=learnable_c).to(device)

    default_curv_mode = config.get('few_curv_mode', 'freeze')
    default_curv_reg = float(config.get('few_curv_reg_lambda', 10.0))
    few_curv_mode = getattr(args, 'few_curv_mode', 'auto')
    if few_curv_mode == 'auto':
        few_curv_mode = default_curv_mode
    args.few_curv_mode = few_curv_mode
    args.few_curv_reg_lambda = float(
        getattr(args, 'few_curv_reg_lambda', default_curv_reg)
        if getattr(args, 'few_curv_reg_lambda', None) is not None
        else default_curv_reg
    )

    pretrained_c_value = curv_param.get().detach()
    lora_curv_param = None

    if args.few:
        if few_curv_mode == 'freeze':
            curv_param.raw_c.requires_grad_(False)
        elif few_curv_mode == 'clone':
            curv_param.raw_c.requires_grad_(False)
            lora_curv_param = CurvatureParam(
                init_c=float(pretrained_c_value.item()),
                min_c=min_c,
                max_c=max_c,
                learnable=True
            ).to(device)
        elif few_curv_mode == 'regularize':
            curv_param.raw_c.requires_grad_(True)
        elif few_curv_mode == 'none':
            pass
        else:
            raise ValueError(f"Unsupported few-shot curvature mode: {few_curv_mode}")

    c_trainable = curv_param.raw_c.requires_grad
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c_status = ": c trainable during transfer" if c_trainable else ": c not trainable during transfer"
    with open("result/GraphLoRA.txt", "a") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"[{timestamp}] New Experiment{c_status}\n")
        f.write(f"Args: {vars(args)}\n")
        f.write(f"Few-shot curvature mode: {few_curv_mode} (lambda={args.few_curv_reg_lambda})\n")
        f.write(f"Config: {config}\n")
        f.write("="*80 + "\n")

    # --- 修改部分：从文件名解析信息并对齐预训练配置 ---
    # 1) 从文件名解析 backbone 是否为双曲
    try:
        parts = args.pretrained_model_name.split('.')
        hyp_part = next(p for p in parts if p.startswith('hyp_'))
        is_hyperbolic_backbone = (hyp_part.split('_')[1] == 'True')
        print(f"Parsed from filename: Pretrained backbone is {'Hyperbolic' if is_hyperbolic_backbone else 'Euclidean'}.")
    except (StopIteration, IndexError) as e:
        raise ValueError(f"Filename '{args.pretrained_model_name}' is not in the expected format. "
                         f"Could not parse hyperbolic status (e.g., 'hyp_True'). Error: {e}")

    # 2) 从文件名解析源数据集（如 'Cora.GRACE....pth' -> 'Cora'），若与 args.pretrain_dataset 不一致则对齐
    try:
        filename_base = os.path.basename(args.pretrained_model_name)
        src_dataset = filename_base.split('.')[0]
        if src_dataset and src_dataset != args.pretrain_dataset:
            print(f"Pretrain dataset overridden by checkpoint: {args.pretrain_dataset} -> {src_dataset}")
            args.pretrain_dataset = src_dataset
            pretrian_datapath = os.path.join('./datasets', args.pretrain_dataset)
            pretrain_dataset = get_dataset(pretrian_datapath, args.pretrain_dataset)[0]
            pretrain_dataset.edge_index = add_remaining_self_loops(pretrain_dataset.edge_index)[0]
            pretrain_dataset = pretrain_dataset.to(device)
    except Exception as e:
        raise ValueError(f"Failed to parse source dataset from '{args.pretrained_model_name}': {e}")

    # 3) 使用解析出的状态初始化 GNN backbone
    gnn = GNN(pretrain_dataset.x.shape[1], config['output_dim'], act(config['activation']),
              config['gnn_type'], config['num_layers'],
              hyperbolic=is_hyperbolic_backbone, curv=curv_param).to(device)

    # 4) 加载由命令行参数指定的模型：过滤掉与模型形状不匹配的权重，避免 size mismatch
    model_path = os.path.join("./pre_trained_gnn/", args.pretrained_model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified pretrained model file does not exist: {model_path}")
        
    state = torch.load(model_path, map_location=device)
    model_state = gnn.state_dict()
    filtered_state = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    missing_keys = [k for k in model_state.keys() if k not in filtered_state]
    unexpected_keys = [k for k in state.keys() if k not in model_state]
    if missing_keys:
        print(f"load_state_dict: skipped {len(missing_keys)} keys due to shape/key mismatch")
    if unexpected_keys:
        print(f"load_state_dict: {len(unexpected_keys)} unexpected keys in checkpoint")
    gnn.load_state_dict(filtered_state, strict=False)
    # --- 修改结束 ---

    gnn.eval()
    for p in gnn.conv.parameters():
        p.requires_grad = False
    # 注意：不要冻结曲率，让它在迁移阶段也能被更新

    # ---- parallel LoRA (hyperbolic fusion) ----
    gnn2 = GNNLoRA(
        pretrain_dataset.x.shape[1],
        config['output_dim'],
        act(config['activation']),
        gnn, config['gnn_type'], config['num_layers'],
        r=args.r,
        hyperbolic=bool(args.hyperbolic_lora),
        lora_alpha=args.lora_alpha,
        curv=curv_param,
        curv_lora=lora_curv_param
    ).to(device)

    # ---- projector / classifier / losses ----
    SMMD = SMMDLoss().to(device)
    projector = Projector(test_dataset.x.shape[1], pretrain_dataset.x.shape[1]).to(device)
    num_classes = int(test_dataset.y.max().item() + 1)
    logreg = LogReg(config['output_dim'], num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # ---- few-shot masks ----
    if args.few:
        # Use enhanced balanced sampling for few-shot
        train_mask, val_mask, test_mask = get_balanced_few_shot_mask(test_dataset, args.shot, args.test_dataset, device)
    else:
        N = test_dataset.x.shape[0]
        index = np.arange(N)
        np.random.shuffle(index)
        train_mask = torch.zeros(N, dtype=torch.bool, device=device)
        val_mask   = torch.zeros(N, dtype=torch.bool, device=device)
        test_mask  = torch.zeros(N, dtype=torch.bool, device=device)
        train_mask[index[:int(N*0.6)]] = True
        val_mask[index[int(N*0.6):int(N*0.8)]] = True
        test_mask[index[int(N*0.8):]] = True

    test_dataset.train_mask = train_mask
    test_dataset.val_mask = val_mask
    test_dataset.test_mask = test_mask

    # Temperature scaling for few-shot calibration
    temp_scale = None
    if args.few:
        temp_scale = TemperatureScaling().to(device)

    ppr_weight = get_ppr_weight(test_dataset)

    # 统计并输出划分规模，便于核验 few/public 模式
    num_train = int(train_mask.sum().item())
    num_val   = int(val_mask.sum().item())
    num_test  = int(test_mask.sum().item())
    split_info = (
        f"Split sizes | train={num_train}, val={num_val}, test={num_test} "
        f"(mode={'few' if args.few else 'public'}, shot={args.shot if args.few else 'N/A'})"
    )
    print(split_info)
    with open("result/GraphLoRA.txt", "a") as f:
        f.write(split_info + "\n")

    # Adaptive loss weights for few-shot learning
    adaptive_weights = get_adaptive_loss_weights(args, num_train)

    # 构造监督图的同类索引 mask
    idx_a, idx_b = torch.tensor([], device=device), torch.tensor([], device=device)
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    train_label = test_dataset.y[train_idx]
    for c in range(num_classes):
        idx_c = train_idx[train_label == c]
        if idx_c.numel() == 0: 
            continue
        idx_a = torch.concat((idx_a, idx_c.repeat_interleave(len(idx_c))))
        idx_b = torch.concat((idx_b, idx_c.repeat(len(idx_c))))
    mask = torch.sparse_coo_tensor(
        indices=torch.stack((idx_a.long(), idx_b.long())),
        values=torch.ones(len(idx_a), device=device),
        size=[test_dataset.x.shape[0], test_dataset.x.shape[0]]
    ).to_dense()
    mask = args.sup_weight * (mask - torch.diag_embed(torch.diag(mask))) + torch.eye(test_dataset.x.shape[0], device=device)

    # ---- optimizer ----
    params_proj   = list(projector.parameters())
    params_logreg = list(logreg.parameters())
    params_gnn2   = [p for n, p in gnn2.named_parameters() if ('raw_c' not in n)]
    curvature_params = []
    if curv_param.raw_c.requires_grad:
        curvature_params.append(curv_param.raw_c)
    if lora_curv_param is not None and lora_curv_param.raw_c.requires_grad:
        curvature_params.append(lora_curv_param.raw_c)

    param_groups = [
        {"params": params_proj,   "lr": args.lr1, "weight_decay": args.wd1},
        {"params": params_logreg, "lr": args.lr2, "weight_decay": args.wd2},
        {"params": params_gnn2,   "lr": args.lr3, "weight_decay": args.wd3},
    ]

    if curvature_params:
        param_groups.append({
            "params": curvature_params,
            "lr": max(args.lr3 * 0.1, 1e-5),
            "weight_decay": 0.0
        })

    # Add temperature scaling parameters for few-shot
    if temp_scale is not None:
        param_groups.append({"params": temp_scale.parameters(), "lr": args.lr2, "weight_decay": 0.0})

    optimizer = torch.optim.Adam(param_groups)

    # Learning rate scheduler for few-shot
    scheduler = None
    if args.few:
        scheduler = get_few_shot_scheduler(optimizer, args.num_epochs)

    pretrain_graph_loader = DataLoader(pretrain_dataset.x, batch_size=32, shuffle=True)

    max_acc, max_test_acc, max_epoch = 0.0, 0.0, 0
    for epoch in range(0, args.num_epochs):
        logreg.train(); projector.train(); gnn2.train()
        if temp_scale is not None:
            temp_scale.train()

        # Feature augmentation for few-shot
        input_features = test_dataset.x
        if args.few:
            input_features = augment_few_shot_features(test_dataset.x, train_mask)

        feature_map = projector(input_features)
        emb, emb1, emb2 = gnn2(feature_map, test_dataset.edge_index, return_euclid=True)

        optimizer.zero_grad()

        smmd_loss_f = batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 32)
        ct_loss = 0.5 * (
            batched_gct_loss(emb1, emb2, 1000, mask, args.tau) +
            batched_gct_loss(emb2, emb1, 1000, mask, args.tau)
        ).mean()

        logits = logreg(emb)

        # Apply temperature scaling for few-shot
        if temp_scale is not None:
            logits = temp_scale(logits)

        train_logits = logits[train_mask]
        train_labels = test_dataset.y[train_mask]
        cls_loss = loss_fn(train_logits, train_labels)

        target_adj = to_dense_adj(test_dataset.edge_index)[0]
        rec_adj = torch.sigmoid(torch.matmul(torch.softmax(logits, dim=1), torch.softmax(logits, dim=1).T))
        pos_weight = float(target_adj.shape[0] * target_adj.shape[0] - test_dataset.edge_index.shape[1]) / max(1, test_dataset.edge_index.shape[1])
        weight_mask = target_adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0), device=device)
        weight_tensor[weight_mask] = pos_weight
        loss_rec = F.binary_cross_entropy(rec_adj.view(-1), target_adj.view(-1), weight=weight_tensor)

        # Use adaptive loss weights
        loss = (adaptive_weights['l1'] * cls_loss +
                adaptive_weights['l2'] * smmd_loss_f +
                adaptive_weights['l3'] * ct_loss +
                adaptive_weights['l4'] * loss_rec)

        curv_reg_component = 0.0
        if args.few and few_curv_mode == 'regularize':
            curv_reg_loss = args.few_curv_reg_lambda * (curv_param.get() - pretrained_c_value).pow(2)
            loss = loss + curv_reg_loss
            curv_reg_component = curv_reg_loss.item()

        loss.backward()
        optimizer.step()

        # Learning rate scheduling for few-shot
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            logreg.eval(); projector.eval(); gnn2.eval()
            if temp_scale is not None:
                temp_scale.eval()

            emb_eval, _, _ = gnn2(projector(test_dataset.x), test_dataset.edge_index, return_euclid=True)
            logits_eval = logreg(emb_eval)

            # Apply temperature scaling for evaluation
            if temp_scale is not None:
                logits_eval = temp_scale(logits_eval)

            pred = logits_eval.argmax(dim=1)
            def acc_of(mask):
                if mask.sum() == 0: return 0.0
                return (pred[mask] == test_dataset.y[mask]).float().mean().item()
            train_acc = acc_of(train_mask)
            val_acc   = acc_of(val_mask)
            test_acc  = acc_of(test_mask)

        c_main_val = float(curv_param.get().detach().item())
        curv_info = f"c={c_main_val:.6f}"
        if lora_curv_param is not None:
            c_lora_val = float(lora_curv_param.get().detach().item())
            curv_info = f"c_main={c_main_val:.6f}, c_lora={c_lora_val:.6f}"
        if args.few and few_curv_mode == 'regularize':
            curv_info = f"{curv_info}, reg={curv_reg_component:.4f}"

        print(f"[Epoch {epoch:04d}] loss={loss.item():.4f} "
              f"cls={cls_loss.item():.4f} smmd={smmd_loss_f.item():.4f} ct={ct_loss.item():.4f} rec={loss_rec.item():.4f} "
              f"| train/val/test={train_acc:.3f}/{val_acc:.3f}/{test_acc:.3f} "
              f"| {curv_info}")

        if val_acc > max_acc:
            max_acc, max_test_acc, max_epoch = val_acc, test_acc, epoch

    print(f"=== Best @ epoch {max_epoch}: val={max_acc:.4f}, test={max_test_acc:.4f} ===")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_label = f"Few({args.shot})" if args.few else "Public"
    result_info = f"{mode_label}, r: {args.r}, {args.pretrain_dataset} to {args.test_dataset}:"
    with open("result/GraphLoRA.txt", "a") as f:
        summary_curv = f"c_main={curv_param.get().detach().item():.6f}"
        if lora_curv_param is not None:
            summary_curv += f", c_lora={lora_curv_param.get().detach().item():.6f}"
        f.write(f"[{timestamp}] {result_info} Best @ epoch {max_epoch}: "
                f"val={max_acc:.4f}, test={max_test_acc:.4f}, {summary_curv} ===\n")
