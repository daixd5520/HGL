#!/usr/bin/env python3
"""Utility script to run the forgetting experiment on multiple GPUs."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch_geometric.utils import add_remaining_self_loops
from yaml import SafeLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.GNN_model import CurvatureParam, GNN
from model.GraphLoRA import LogReg
from util import act, get_dataset, lorentz_logmap0

AVAILABLE_DATASETS = ["Cora", "CiteSeer", "PubMed", "Photo", "Computers"]
SOURCE_RE = re.compile(r"SourceEval\s+([A-Za-z0-9_]+)<-([A-Za-z0-9_]+):\s+test=([0-9]*\.?[0-9]+)")
BEST_RE = re.compile(
    r"Best\s*@\s*epoch\s*(\d+):\s*val\s*=\s*([0-9]*\.?[0-9]+),\s*test\s*=\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run forgetting experiments across multiple GPUs")
    parser.add_argument("--pretrained-model", required=True, help="Path to the pretrained checkpoint (.pth)")
    parser.add_argument(
        "--d2",
        nargs="+",
        help="Target datasets (d2) to fine-tune on. Defaults to all datasets except the source dataset.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="List of GPU ids to use (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./experiments/forgetting",
        help="Directory to store logs and aggregated CSV",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to the main configuration file (used for model definitions)",
    )
    parser.add_argument(
        "--para-config",
        type=str,
        default="./config2.yaml",
        help="Path to the transfer hyper-parameter configuration file (forwarded to main.py)",
    )
    parser.add_argument(
        "--direct-device",
        type=str,
        default="cpu",
        help="Device for direct evaluation (default: cpu)",
    )
    parser.add_argument(
        "--direct-epochs",
        type=int,
        default=200,
        help="Number of epochs for the direct logistic regression evaluation",
    )
    return parser.parse_args()


def ensure_pretrained_available(pretrained_path: Path) -> str:
    pretrained_path = pretrained_path.expanduser().resolve()
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Cannot find pretrained checkpoint: {pretrained_path}")

    dest_dir = Path("./pre_trained_gnn")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / pretrained_path.name

    if pretrained_path != dest_path.resolve():
        if not dest_path.exists() or pretrained_path.stat().st_mtime > dest_path.stat().st_mtime:
            shutil.copy2(pretrained_path, dest_path)
    return pretrained_path.name


def infer_source_dataset(checkpoint_name: str) -> str:
    stem = checkpoint_name.split(".")[0]
    if stem not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unable to infer source dataset from checkpoint name '{checkpoint_name}'. "
            f"Expected prefix to be one of {AVAILABLE_DATASETS}."
        )
    return stem


def load_config_for_dataset(config_path: str, dataset: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader)
    if dataset not in config:
        raise KeyError(f"Dataset '{dataset}' not found in config file '{config_path}'.")
    return config[dataset]


def prepare_masks(data, device, seed: int = 0):
    train_mask = getattr(data, "train_mask", None)
    val_mask = getattr(data, "val_mask", None)
    test_mask = getattr(data, "test_mask", None)

    if train_mask is not None and val_mask is not None and test_mask is not None:
        train_mask = train_mask.to(device).bool()
        val_mask = val_mask.to(device).bool()
        test_mask = test_mask.to(device).bool()
        if train_mask.numel() == data.x.shape[0]:
            return train_mask, val_mask, test_mask

    num_nodes = data.x.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    train_cut = int(num_nodes * 0.6)
    val_cut = int(num_nodes * 0.8)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_cut]] = True
    val_mask[perm[train_cut:val_cut]] = True
    test_mask[perm[val_cut:]] = True

    return train_mask.to(device), val_mask.to(device), test_mask.to(device)


def train_logreg(features, labels, train_mask, val_mask, test_mask, device, epochs: int = 200):
    classifier = LogReg(features.size(1), int(labels.max().item() + 1)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_test = 0.0

    for _ in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            logits_eval = classifier(features)
            preds = logits_eval.argmax(dim=1)

            def _acc(mask):
                if mask.sum() == 0:
                    return 0.0
                return (preds[mask] == labels[mask]).float().mean().item()

            val_acc = _acc(val_mask)
            test_acc = _acc(test_mask)
            if val_acc >= best_val:
                best_val = val_acc
                best_test = test_acc

    return best_test


def direct_evaluation(pretrained_name: str, dataset: str, config_path: str, device: str, epochs: int) -> float:
    cfg = load_config_for_dataset(config_path, dataset)
    dataset_path = Path("./datasets") / dataset
    data = get_dataset(str(dataset_path), dataset)[0]
    data.edge_index = add_remaining_self_loops(data.edge_index)[0]

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    torch_device = torch.device(device)
    data = data.to(torch_device)

    curv_param = CurvatureParam(
        init_c=float(cfg.get("curvature", 1.0)),
        min_c=float(cfg.get("min_curvature", 1e-4)),
        max_c=float(cfg.get("max_curvature", 10.0)),
        learnable=bool(cfg.get("learnable_curvature", True)),
    ).to(torch_device)

    gnn = GNN(
        data.x.shape[1],
        int(cfg["output_dim"]),
        act(cfg["activation"]),
        cfg["gnn_type"],
        int(cfg["num_layers"]),
        hyperbolic=bool(cfg.get("hyperbolic_backbone", True)),
        curv=curv_param,
    ).to(torch_device)

    checkpoint_path = Path("./pre_trained_gnn") / pretrained_name
    state = torch.load(checkpoint_path, map_location=torch_device)
    model_state = gnn.state_dict()
    filtered_state = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
    gnn.load_state_dict(filtered_state, strict=False)
    gnn.eval()

    with torch.no_grad():
        outputs = gnn(data.x, data.edge_index)
        if bool(cfg.get("hyperbolic_backbone", True)):
            features = lorentz_logmap0(outputs, float(curv_param.get().item()))
        else:
            features = outputs

    features = features.detach()
    labels = data.y.to(torch_device)
    train_mask, val_mask, test_mask = prepare_masks(data, torch_device)

    return train_logreg(features, labels, train_mask, val_mask, test_mask, torch_device, epochs=epochs)


def run_transfer(dataset: str, gpu_id: int, args: argparse.Namespace, exp_dir: Path) -> Dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        os.sys.executable,
        "main.py",
        "--is_transfer",
        "True",
        "--few",
        "False",
        "--pretrain_dataset",
        args.source_dataset,
        "--test_dataset",
        dataset,
        "--pretrained_model_name",
        args.pretrained_name,
        "--eval_on_source",
        "True",
        "--gpu_id",
        "0",
        "--config",
        args.config,
        "--para_config",
        args.para_config,
    ]

    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    stdout_path = exp_dir / f"{dataset}_stdout.txt"
    stderr_path = exp_dir / f"{dataset}_stderr.txt"
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")

    if process.returncode != 0:
        raise RuntimeError(
            f"Transfer run for dataset '{dataset}' failed with return code {process.returncode}. "
            f"See {stdout_path} and {stderr_path} for details."
        )

    source_match = SOURCE_RE.search(process.stdout)
    if not source_match:
        raise RuntimeError(
            f"Could not find source evaluation output in stdout for dataset '{dataset}'. "
            f"Make sure --eval_on_source is enabled."
        )

    _, _, source_acc = source_match.groups()
    best_match: Optional[re.Match] = None
    for match in BEST_RE.finditer(process.stdout):
        best_match = match

    target_acc = float(best_match.group(3)) if best_match else None

    return {
        "dataset": dataset,
        "gpu_id": gpu_id,
        "source_acc": float(source_acc),
        "target_acc": target_acc,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def main():
    args = parse_args()

    if not args.gpus:
        raise ValueError("Please specify at least one GPU id via --gpus.")

    pretrained_name = ensure_pretrained_available(Path(args.pretrained_model))
    source_dataset = infer_source_dataset(pretrained_name)

    if args.d2:
        target_datasets = [d for d in args.d2 if d != source_dataset]
    else:
        target_datasets = [d for d in AVAILABLE_DATASETS if d != source_dataset]

    invalid = [d for d in target_datasets if d not in AVAILABLE_DATASETS]
    if invalid:
        raise ValueError(f"Unknown datasets specified for --d2: {invalid}. Available: {AVAILABLE_DATASETS}")

    if not target_datasets:
        raise ValueError("No target datasets to run. After removing the source dataset the list is empty.")

    out_dir = Path(args.out)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = out_dir / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FORGETTING] Source checkpoint: {pretrained_name} (dataset {source_dataset})")
    print(f"[FORGETTING] Target datasets: {target_datasets}")
    print(f"[FORGETTING] Logs will be saved to: {exp_dir}")

    args.pretrained_name = pretrained_name
    args.source_dataset = source_dataset

    print("[FORGETTING] Running direct evaluation on source dataset (no transfer)...")
    direct_acc = direct_evaluation(pretrained_name, source_dataset, args.config, args.direct_device, args.direct_epochs)
    print(f"[FORGETTING] Direct evaluation accuracy on {source_dataset}: {direct_acc:.4f}")

    gpu_cycle = (args.gpus * ((len(target_datasets) // max(1, len(args.gpus))) + 1))[: len(target_datasets)]

    results = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = []
        for dataset, gpu_id in zip(target_datasets, gpu_cycle):
            futures.append(executor.submit(run_transfer, dataset, gpu_id, args, exp_dir))

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(
                f"[FORGETTING] Finished transfer {args.source_dataset}->{res['dataset']} on GPU {res['gpu_id']}: "
                f"source_acc={res['source_acc']:.4f}"
            )

    results.sort(key=lambda x: target_datasets.index(x["dataset"]))

    matrix = {"direct": direct_acc}
    for res in results:
        matrix[res["dataset"]] = res["source_acc"]

    df = pd.DataFrame([matrix], index=[source_dataset])
    csv_path = exp_dir / "forgetting_matrix.csv"
    df.to_csv(csv_path)

    detail_path = exp_dir / "runs.jsonl"
    with detail_path.open("w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    meta = {
        "timestamp": timestamp,
        "pretrained_model": pretrained_name,
        "source_dataset": source_dataset,
        "target_datasets": target_datasets,
        "direct_acc": direct_acc,
        "config": args.config,
        "para_config": args.para_config,
        "gpus": args.gpus,
        "command": "python " + " ".join(os.sys.argv),
    }
    (exp_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[FORGETTING] Aggregated matrix saved to: {csv_path}")


if __name__ == "__main__":
    main()

