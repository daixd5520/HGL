"""Utility script to pretrain on multiple source datasets and evaluate on one target.

Example:
    python experiments/multi_source_pretrain_eval.py \
        --pretrain_datasets PubMed,CiteSeer \
        --target_dataset Squirrel \
        --config ./config.yaml \
        --para_config ./config2.yaml \
        --gpu_id 0

Each source dataset is pretrained separately. The produced checkpoints are
immediately used for transfer learning on the target dataset so you can compare
cross-dataset performance.
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import List

import torch
import yaml
from yaml import SafeLoader

from model.GraphLoRA import transfer
from pre_train import pretrain
from util import get_parameter


def _parse_dataset_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(',') if item.strip()]


def _str2bool(val):
    if isinstance(val, bool):
        return val
    lowered = str(val).strip().lower()
    if lowered in {'true', 't', 'yes', 'y', '1'}:
        return True
    if lowered in {'false', 'f', 'no', 'n', '0'}:
        return False
    raise argparse.ArgumentTypeError('Boolean value expected (e.g., True/False).')


def _select_device(gpu_id: int) -> int:
    """Mirror the main entry point's GPU selection logic."""
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        if visible == 1:
            torch.cuda.set_device(0)
            return 0
        if visible > 1:
            chosen = max(0, min(gpu_id, visible - 1))
            torch.cuda.set_device(chosen)
            return chosen
    return gpu_id


def _build_transfer_args(base_args: argparse.Namespace, *, checkpoint_path: str, pretrain_dataset: str,
                          curvature: float, min_curvature: float, max_curvature: float) -> SimpleNamespace:
    return SimpleNamespace(
        pretrain_dataset=pretrain_dataset,
        test_dataset=base_args.target_dataset,
        gpu_id=base_args.gpu_id,
        pretext=base_args.pretext,
        config=base_args.config,
        para_config=base_args.para_config,
        pretrain_curvature=None,
        pretrain_run_tag=None,
        pretrain_output_name=None,
        is_pretrain=False,
        is_transfer=True,
        is_reduction=base_args.is_reduction,
        few=base_args.few,
        shot=base_args.shot,
        tau=base_args.tau,
        sup_weight=base_args.sup_weight,
        r=base_args.r,
        hyperbolic_lora=base_args.hyperbolic_lora,
        curvature=curvature,
        min_curvature=min_curvature,
        max_curvature=max_curvature,
        lora_alpha=base_args.lora_alpha,
        pretrained_model_name=os.path.basename(checkpoint_path),
    )


def run_multi_source_experiment(args: argparse.Namespace) -> None:
    config_all = yaml.load(open(args.config), Loader=SafeLoader)
    transfer_config = config_all['transfer']

    results = []
    for src_dataset in args.pretrain_datasets:
        if src_dataset not in config_all:
            raise KeyError(f"{src_dataset} not found in {args.config} for pretraining config")

        print(f"\n=== Pretraining on {src_dataset} ===")
        pretrain_cfg = config_all[src_dataset]
        curvature = float(pretrain_cfg.get('curvature', args.curvature))
        min_c = float(pretrain_cfg.get('min_curvature', args.min_curvature))
        max_c = float(pretrain_cfg.get('max_curvature', args.max_curvature))

        ckpt_path = pretrain(
            src_dataset,
            args.pretext,
            pretrain_cfg,
            args.gpu_id,
            args.is_reduction,
            init_curvature=curvature,
            run_tag=args.pretrain_run_tag,
        )
        print(f"[Experiment] Pretrained checkpoint stored at {ckpt_path}")

        transfer_args = _build_transfer_args(
            args,
            checkpoint_path=ckpt_path,
            pretrain_dataset=src_dataset,
            curvature=curvature,
            min_curvature=min_c,
            max_curvature=max_c,
        )
        transfer_args = get_parameter(transfer_args)

        print(f"\n--- Transfer: {src_dataset} -> {args.target_dataset} ---")
        transfer(transfer_args, transfer_config, args.gpu_id, args.is_reduction)
        results.append((src_dataset, ckpt_path))

    print("\n==== Summary ====")
    for src_dataset, ckpt_path in results:
        print(f"Pretrained on {src_dataset} using {ckpt_path}; evaluated on {args.target_dataset}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrain_datasets', type=_parse_dataset_list, default='PubMed,CiteSeer')
    parser.add_argument('--target_dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--pretext', type=str, default='GRACE')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--para_config', type=str, default='./config2.yaml')
    parser.add_argument('--pretrain_run_tag', type=str, default=None,
                        help='Optional tag appended to pretrained checkpoint names.')
    parser.add_argument('--is_reduction', type=_str2bool, default=True)
    parser.add_argument('--few', type=_str2bool, default=False)
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--sup_weight', type=float, default=0.2)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--hyperbolic_lora', type=_str2bool, default=True)
    parser.add_argument('--curvature', type=float, default=1.0)
    parser.add_argument('--min_curvature', type=float, default=1e-4)
    parser.add_argument('--max_curvature', type=float, default=10.0)
    parser.add_argument('--lora_alpha', type=float, default=16.0)

    parsed = parser.parse_args()
    parsed.pretrain_datasets = _parse_dataset_list(parsed.pretrain_datasets) if isinstance(parsed.pretrain_datasets, str) else parsed.pretrain_datasets
    parsed.gpu_id = _select_device(parsed.gpu_id)

    run_multi_source_experiment(parsed)
