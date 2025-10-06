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
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--pretext', type=str, default='GRACE')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--para_config', type=str, default='./config2.yaml')
    # 兼容 "--few True/False" 的布尔解析
    def str2bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ('yes', 'true', 't', 'y', '1'):
            return True
        if s in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected for --few (e.g., True/False).')

    parser.add_argument('--is_pretrain', type=str2bool, default=False)
    parser.add_argument('--is_transfer', type=str2bool, default=True)
    parser.add_argument('--is_reduction', type=str2bool, default=True)

    def str2bool_duplicate(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ('yes', 'true', 't', 'y', '1'):
            return True
        if s in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected for --few (e.g., True/False).')

    parser.add_argument('--few', type=str2bool, default=False)
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--sup_weight', type=float, default=0.2)
    parser.add_argument('--r', type=int, default=32)
    ###Three new parameters for hyperbolic LoRA
    parser.add_argument('--hyperbolic_lora', type=bool, default=True)
    parser.add_argument('--curvature', type=float, default=1.0)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    
    # --- 新增参数：指定预训练模型文件名 ---
    parser.add_argument('--pretrained_model_name', type=str, default=None,
                        help='Filename of the pretrained model to load for transfer learning (e.g., "PubMed.GRACE.GAT.hyp_True.False.20250910-013000.pth")')
    # --- 修改结束 ---

    parser.add_argument('--eval_on_source', type=str2bool, default=False,
                        help='Whether to evaluate the fine-tuned model on the source dataset (d1) after transfer.')
    
    # ###learnable curvature
    # parser.add_argument('--learnable_curvature', type=bool, default=False)
    # parser.add_argument('--k_init', type=float, default=None)
    args = parser.parse_args()
    args = get_parameter(args)

    # 更稳妥的单卡/多卡选择：若仅一张可见卡则强制使用 0
    if torch.cuda.is_available():
        num_visible = torch.cuda.device_count()
        if num_visible == 0:
            pass
        elif num_visible == 1:
            args.gpu_id = 0
            torch.cuda.set_device(0)
        else:
            args.gpu_id = max(0, min(args.gpu_id, num_visible - 1))
            torch.cuda.set_device(args.gpu_id)

    if args.is_pretrain:
        config_pretrain = yaml.load(open(args.config), Loader=SafeLoader)[args.pretrain_dataset]
        pretrain(args.pretrain_dataset, args.pretext, config_pretrain, args.gpu_id, args.is_reduction)
    
    if args.is_transfer:
        # --- 修改部分：增加检查，确保在迁移时提供了模型文件名 ---
        if args.pretrained_model_name is None:
            raise ValueError("For transfer learning (`--is_transfer=True`), you must specify a model file using `--pretrained_model_name`.")
        # --- 修改结束 ---
        config_transfer = yaml.load(open(args.config), Loader=SafeLoader)['transfer']
        transfer(args, config_transfer, args.gpu_id, args.is_reduction)
