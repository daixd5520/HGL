import yaml
from yaml import SafeLoader
import argparse
import torch
from pre_train import pretrain
from model.GraphLoRA import transfer
from util import get_parameter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_dataset', type=str, default='PubMed')
    parser.add_argument('--test_dataset', type=str, default='CiteSeer')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--pretext', type=str, default='GRACE')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--para_config', type=str, default='./config2.yaml')
    parser.add_argument('--is_pretrain', type=bool, default=False)
    parser.add_argument('--is_transfer', type=bool, default=True)
    parser.add_argument('--is_reduction', type=bool, default=True)
    parser.add_argument('--few', type=bool, default=False)
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--sup_weight', type=float, default=0.2)
    parser.add_argument('--r', type=int, default=32)
    args = parser.parse_args()
    args = get_parameter(args)

    assert args.gpu_id in range(0, 2)
    torch.cuda.set_device(args.gpu_id)

    if args.is_pretrain:
        config_pretrain = yaml.load(open(args.config), Loader=SafeLoader)[args.pretrain_dataset]
        pretrain(args.pretrain_dataset, args.pretext, config_pretrain, args.gpu_id, args.is_reduction)
    
    if args.is_transfer:
        config_transfer = yaml.load(open(args.config), Loader=SafeLoader)['transfer']
        transfer(args, config_transfer, args.gpu_id, args.is_reduction)
