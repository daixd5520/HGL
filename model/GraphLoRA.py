from model.GNN_model import GNN, GNNLoRA, HyperbolicLoRA
import torch
import torch.nn as nn
import os
from torch_geometric.transforms import SVDFeatureReduction
from util import get_dataset, act, SMMDLoss, mkdir, get_ppr_weight
from util import get_few_shot_mask, batched_smmd_loss, batched_gct_loss
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader


class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.initialize()

    def forward(self, x):
        return self.fc(x)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x):
        return self.fc(x)
    
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


def transfer(args, config, gpu_id, is_reduction):
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # load data
    pretrian_datapath = os.path.join('./datasets', args.pretrain_dataset)
    test_datapath = os.path.join('./datasets', args.test_dataset)
    pretrain_dataset = get_dataset(pretrian_datapath, args.pretrain_dataset)[0]
    test_dataset = get_dataset(test_datapath, args.test_dataset)[0]
    if is_reduction:
        feature_reduce = SVDFeatureReduction(out_channels=100)
        pretrain_dataset = feature_reduce(pretrain_dataset)
        test_dataset = feature_reduce(test_dataset)
    pretrain_dataset.edge_index = add_remaining_self_loops(pretrain_dataset.edge_index)[0]
    test_dataset.edge_index = add_remaining_self_loops(test_dataset.edge_index)[0]
    pretrain_dataset = pretrain_dataset.to(device)
    test_dataset = test_dataset.to(device)

    # target adj
    target_adj = to_dense_adj(test_dataset.edge_index)[0]
    pos_weight = float(test_dataset.x.shape[0] * test_dataset.x.shape[0] - test_dataset.edge_index.shape[1]) / test_dataset.edge_index.shape[1]
    weight_mask = target_adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight / 10

    gnn = GNN(pretrain_dataset.x.shape[1], config['output_dim'], act(config['activation']), config['gnn_type'], config['num_layers'])
    model_path = "./pre_trained_gnn/{}.{}.{}.{}.pth".format(args.pretrain_dataset, args.pretext, config['gnn_type'], args.is_reduction)
    gnn.load_state_dict(torch.load(model_path))
    gnn.to(device)
    gnn.eval()
    for param in gnn.conv.parameters():
        param.requires_grad = False

    # gnn2 = GNNLoRA(pretrain_dataset.x.shape[1], config['output_dim'], act(config['activation']), gnn, config['gnn_type'], config['num_layers'], r=args.r)
    
    gnn2 = GNNLoRA(
        pretrain_dataset.x.shape[1],
        config['output_dim'],
        act(config['activation']),
        gnn, config['gnn_type'], config['num_layers'],
        r=args.r,
        hyperbolic=args.hyperbolic_lora,
        lora_alpha=args.lora_alpha,
        curvature=args.curvature
    )
    
    def print_lora_summary(model):
        import torch.nn as nn
        from model.GNN_model import HyperbolicLoRA  # 确保路径一致

        h_count, e_count = 0, 0
        for m in model.modules():
            if isinstance(m, HyperbolicLoRA):
                h_count += 1
            # 欧氏 LoRA 的特征：nn.Sequential(Linear(in, r), Linear(r, out))
            if isinstance(m, nn.Sequential) and len(m) == 2:
                if all(hasattr(m[i], "weight") for i in (0, 1)):
                    e_count += 1

        print(f"[LoRA summary] HyperbolicLoRA: {h_count}, Euclidean-Sequential: {e_count}")
        for n, p in model.named_parameters():
            if "A.weight" in n or "B.weight" in n:
                print(f"  {n}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")

    # 例：你的 LoRA 分支是 gnn2
    print_lora_summary(gnn2)
    
    gnn2.to(device)
    
    def attach_hyperbolic_hooks(model, every=200):
        step = {"i": 0}
        def hook(mod, inp, out):
            step["i"] += 1
            if step["i"] % every == 0:
                x = inp[0].detach()
                print(f"[Hook/HyperbolicLoRA] step={step['i']}, in_norm={x.norm(dim=-1).mean().item():.4f}, out_norm={out.detach().norm(dim=-1).mean().item():.4f}, c={mod.c}")
        for m in model.modules():
            if isinstance(m, HyperbolicLoRA):
                m.register_forward_hook(hook)

    attach_hyperbolic_hooks(gnn2, every=100)
    
    
    gnn2.train()

    SMMD = SMMDLoss().to(device)

    projector = Projector(test_dataset.x.shape[1], pretrain_dataset.x.shape[1])
    projector = projector.to(device)
    projector.train()

    # optimizer
    logreg = LogReg(config['output_dim'], max(test_dataset.y) + 1)
    logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

    if args.test_dataset in ['PubMed', 'CiteSeer', 'Cora']:
        if args.few:
            train_mask, val_mask, test_mask = get_few_shot_mask(test_dataset, args.shot, args.test_dataset, device)
        else:
            train_mask = test_dataset.train_mask
            val_mask = test_dataset.val_mask
            test_mask = test_dataset.test_mask
    else:
        if args.few:
            train_mask, val_mask, test_mask = get_few_shot_mask(test_dataset, args.shot, args.test_dataset, device)
        else:
            index = np.arange(test_dataset.x.shape[0])
            np.random.shuffle(index)
            train_mask = torch.zeros(test_dataset.x.shape[0]).bool().to(device)
            val_mask = torch.zeros(test_dataset.x.shape[0]).bool().to(device)
            test_mask = torch.zeros(test_dataset.x.shape[0]).bool().to(device)
            train_mask[index[:int(len(index) * 0.1)]] = True
            val_mask[index[int(len(index) * 0.1):int(len(index) * 0.2)]] = True
            test_mask[index[int(len(index) * 0.2):]] = True
    mask = torch.zeros((test_dataset.x.shape[0], test_dataset.x.shape[0])).to(device)
    ppr_weight = get_ppr_weight(test_dataset)
    idx_a = torch.tensor([]).to(device)
    idx_b = torch.tensor([]).to(device)
    for i in range(max(test_dataset.y) + 1):
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        train_label = test_dataset.y[train_idx]
        idx_a = torch.concat((idx_a, train_idx[train_label == i].repeat_interleave(len(train_idx[train_label == i]))))
        idx_b = torch.concat((idx_b, train_idx[train_label == i].repeat(len(train_idx[train_label == i]))))
    mask = torch.sparse_coo_tensor(indices=torch.stack((idx_a, idx_b)), values=torch.ones(len(idx_a)).to(device), size=[test_dataset.x.shape[0], test_dataset.x.shape[0]]).to_dense()
    mask = args.sup_weight * (mask - torch.diag_embed(torch.diag(mask))) + torch.eye(test_dataset.x.shape[0]).to(device)
    optimizer = torch.optim.Adam([{"params": projector.parameters(), 'lr': args.lr1, 'weight_decay': args.wd1}, {"params": logreg.parameters(), 'lr': args.lr2, 'weight_decay': args.wd2}, {"params": gnn2.parameters(), 'lr': args.lr3, 'weight_decay': args.wd3}])

    test_dataset.train_mask = train_mask
    test_dataset.val_mask = val_mask
    test_dataset.test_mask = test_mask

    train_labels = test_dataset.y[train_mask]
    val_labels = test_dataset.y[val_mask]
    test_labels = test_dataset.y[test_mask]

    pretrain_graph_loader = DataLoader(pretrain_dataset.x, batch_size=128, shuffle=True)
    max_acc = 0
    max_test_acc = 0
    max_epoch = 0

    for epoch in range(0, args.num_epochs):
        logreg.train()
        projector.train()
  
        pos_weight = float(target_adj.shape[0] * target_adj.shape[0] - target_adj.sum()) / target_adj.sum()
        weight_mask = target_adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(device)
        weight_tensor[weight_mask] = pos_weight

        feature_map = projector(test_dataset.x)
        emb, emb1, emb2 = gnn2(feature_map, test_dataset.edge_index)
        train_labels = test_dataset.y[train_mask]
        optimizer.zero_grad()

        smmd_loss_f = batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)      
        ct_loss = 0.5 * (batched_gct_loss(emb1, emb2, 1000, mask, args.tau) + batched_gct_loss(emb2, emb1, 1000, mask, args.tau)).mean()
        logits = logreg(emb)
        train_logits = logits[train_mask]

        rec_adj = torch.sigmoid(torch.matmul(torch.softmax(logits, dim=1), torch.softmax(logits, dim=1).T))
        loss_rec = F.binary_cross_entropy(rec_adj.view(-1), target_adj.view(-1), weight=weight_tensor)

        preds = torch.argmax(train_logits, dim=1)
        cls_loss = loss_fn(train_logits, train_labels)
        loss = args.l1 * cls_loss + args.l2 * smmd_loss_f +  args.l3 * ct_loss + args.l4 * loss_rec
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        logreg.eval()
        projector.eval()
        with torch.no_grad():
            val_logits = logits[val_mask]
            test_logits = logits[test_mask]
            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)
            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]
            print('Epoch: {}, train_acc: {:.4f}, val_acc: {:4f}, test_acc: {:4f}'.format(epoch, train_acc, val_acc, test_acc))
            if max_acc < val_acc:
                max_acc = val_acc
                max_test_acc = test_acc
                max_epoch = epoch + 1
    print('epoch: {}, val_acc: {:4f}, test_acc: {:4f}'.format(max_epoch, max_acc, max_test_acc))
    result_path = './result'
    mkdir(result_path)
    if args.few:
        with open(result_path + '/GraphLoRA.txt', 'a') as f:
            f.write('Few: True, r: %d, Shot: %d, %s to %s: val_acc: %f, test_acc: %f\n'%(args.r, args.shot, args.pretrain_dataset, args.test_dataset, max_acc, max_test_acc))
    else:
        with open(result_path + '/GraphLoRA.txt', 'a') as f:
            f.write('Few: False, r: %d, %s to %s: val_acc: %f, test_acc: %f\n'%(args.r, args.pretrain_dataset, args.test_dataset, max_acc, max_test_acc))
