from model.GNN_model import GNN
from model.GRACE_model import GRACE
from time import time
import torch
import os
from util import get_dataset, act, mkdir
from torch_geometric.transforms import SVDFeatureReduction


def pretrain(dataname, pretext, config, gpu, is_reduction=False):
    print(os.getcwd())
    path = os.path.join('./datasets', dataname)
    dataset = get_dataset(path, dataname)
    data = dataset[0]
    if is_reduction:
        feature_reduce = SVDFeatureReduction(out_channels=100)
        data = feature_reduce(data)
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    pre_trained_model_path = './pre_trained_gnn/'
    mkdir(pre_trained_model_path)
    print("create PreTrain instance...")
    input_dim = data.x.shape[1]
    output_dim = config['output_dim']
    num_proj_dim = config['num_proj_dim']
    activation = act(config['activation'])
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']
    tau = config['tau']
    gnn_type = config['gnn_type']
    num_layers = config['num_layers']
    drop_edge_rate = config['drop_edge_rate']
    drop_feature_rate = config['drop_feature_rate']
    gnn = GNN(input_dim, output_dim, activation, gnn_type, num_layers)
    if pretext == 'GRACE':
        pretrain_model = GRACE(gnn, output_dim, num_proj_dim, drop_edge_rate, drop_feature_rate, tau)
    else:
        pretrain_model = GRACE(gnn, output_dim, num_proj_dim, drop_edge_rate, drop_feature_rate, tau)
    pretrain_model.to(device)
    print("pre-training...")
    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time()
    prev = start
    pretrain_model.train()
    min_loss = 100000
    model_path = pre_trained_model_path + "{}.{}.{}.{}.pth".format(dataname, pretext, gnn_type, is_reduction)
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss = pretrain_model.compute_loss(data.x, data.edge_index)
        loss.backward()
        optimizer.step()
        now = time()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
        if min_loss > loss:
            min_loss = loss
            torch.save(pretrain_model.gnn.state_dict(), model_path)
            print("+++model saved ! {}.{}.{}.pth".format(dataname, pretext, gnn_type))
    print("=== Final ===")