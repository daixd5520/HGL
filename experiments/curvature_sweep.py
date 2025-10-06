#!/usr/bin/env python3
"""Sweep the initial curvature for pre-training and evaluate transfer performance."""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pandas as pd
import yaml
from numpy import mean as np_mean
from numpy import std as np_std
from yaml import SafeLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pre_train import pretrain
from experiments.forgetting_experiment import (  # pylint: disable=wrong-import-position
    AVAILABLE_DATASETS,
    direct_evaluation,
    run_transfer,
)


DEFAULT_CURVATURES = [0.1, 0.3, 1.0, 3.0, 10.0]


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"yes", "true", "t", "y", "1"}:
        return True
    if text in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (e.g., True/False).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run curvature sweep experiments with automated transfer evaluation")
    parser.add_argument("--dataset", required=True, help="Source dataset used for pre-training (d1)")
    parser.add_argument(
        "--targets",
        nargs="+",
        help="Target datasets to transfer onto. Defaults to all other AVAILABLE_DATASETS.",
    )
    parser.add_argument(
        "--curvatures",
        nargs="+",
        type=float,
        default=DEFAULT_CURVATURES,
        help=(
            "Initial curvature values to sweep. Defaults to a log-spaced range that covers flatter "
            "(0.1) through highly curved (10.0) manifolds."
        ),
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of repetitions per curvature setting")
    parser.add_argument("--pretext", type=str, default="GRACE", help="Pre-training objective to use")
    parser.add_argument("--pretrain-gpu", type=int, default=0, help="GPU id used for pre-training runs")
    parser.add_argument(
        "--transfer-gpus",
        nargs="+",
        type=int,
        default=[0],
        help="List of GPU ids to cycle through for transfer runs",
    )
    parser.add_argument("--config", type=str, default="./config.yaml", help="Main configuration file")
    parser.add_argument("--para-config", type=str, default="./config2.yaml", help="Transfer hyper-parameters")
    parser.add_argument("--out", type=str, default="./experiments/curvature", help="Directory for experiment outputs")
    parser.add_argument(
        "--direct-device",
        type=str,
        default="cpu",
        help="Device for direct evaluation of the frozen encoder",
    )
    parser.add_argument(
        "--direct-epochs",
        type=int,
        default=200,
        help="Epochs for the logistic regression probe during direct evaluation",
    )
    parser.add_argument(
        "--is-reduction",
        type=str2bool,
        default=False,
        help="Whether to enable feature reduction in pre-training",
    )
    return parser.parse_args()


def prepare_dataset_config(base_cfg: Dict, curvature: float) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["curvature"] = float(curvature)
    min_c = float(cfg.get("min_curvature", curvature))
    max_c = float(cfg.get("max_curvature", curvature))
    if curvature < min_c:
        cfg["min_curvature"] = curvature
    if curvature > max_c:
        cfg["max_curvature"] = curvature
    return cfg


def main() -> None:
    args = parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if not args.transfer_gpus:
        raise ValueError("Please specify at least one GPU id for --transfer-gpus")

    with open(args.config, "r", encoding="utf-8") as f:
        raw_config = yaml.load(f, Loader=SafeLoader)

    if args.dataset not in raw_config:
        raise KeyError(f"Dataset '{args.dataset}' not found in config file '{args.config}'.")
    base_cfg = raw_config[args.dataset]

    available_targets = [d for d in AVAILABLE_DATASETS if d != args.dataset]
    if args.targets:
        target_datasets = [d for d in args.targets if d != args.dataset]
    else:
        target_datasets = available_targets

    invalid = [d for d in target_datasets if d not in AVAILABLE_DATASETS]
    if invalid:
        raise ValueError(
            f"Unknown targets specified: {invalid}. Available options: {AVAILABLE_DATASETS}."
        )
    if not target_datasets:
        raise ValueError("No valid target datasets were provided for transfer evaluation.")

    curvatures = [float(c) for c in args.curvatures]
    if any(c <= 0 for c in curvatures):
        raise ValueError("All curvature values must be positive.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_root = Path(args.out).expanduser().resolve() / f"{args.dataset}_curvature_{timestamp}"
    exp_root.mkdir(parents=True, exist_ok=True)

    print(f"[CURVATURE] Source dataset: {args.dataset}")
    print(f"[CURVATURE] Target datasets: {target_datasets}")
    print(f"[CURVATURE] Curvature sweep: {curvatures}")
    print(f"[CURVATURE] Outputs will be saved to: {exp_root}")

    column_order = ["direct"] + target_datasets
    summary_rows: Dict[str, Dict[str, str]] = {}
    run_records: List[Dict] = []

    for curvature in curvatures:
        print(f"[CURVATURE] === Sweeping initial curvature c={curvature:.4g} ===")
        aggregated: Dict[str, List[float]] = {key: [] for key in column_order}
        curvature_dir = exp_root / f"curv_{curvature:.4g}"
        curvature_dir.mkdir(parents=True, exist_ok=True)

        for repeat_idx in range(1, args.repeats + 1):
            print(
                f"[CURVATURE] >> Repeat {repeat_idx}/{args.repeats} for c={curvature:.4g}: starting pre-training..."
            )
            cfg = prepare_dataset_config(base_cfg, curvature)
            model_tag = f"c{curvature:.4g}_r{repeat_idx}"
            model_name = pretrain(
                args.dataset,
                args.pretext,
                cfg,
                args.pretrain_gpu,
                args.is_reduction,
                model_tag=model_tag,
            )

            run_records.append(
                {
                    "type": "pretrain",
                    "dataset": args.dataset,
                    "curvature": curvature,
                    "repeat": repeat_idx,
                    "model_name": model_name,
                    "gpu_id": args.pretrain_gpu,
                    "model_path": str(Path("./pre_trained_gnn") / model_name),
                }
            )

            direct_acc = direct_evaluation(
                model_name,
                args.dataset,
                args.config,
                args.direct_device,
                args.direct_epochs,
            )
            aggregated["direct"].append(direct_acc)
            run_records.append(
                {
                    "type": "direct",
                    "dataset": args.dataset,
                    "curvature": curvature,
                    "repeat": repeat_idx,
                    "accuracy": direct_acc,
                    "device": args.direct_device,
                    "epochs": args.direct_epochs,
                    "model_name": model_name,
                }
            )

            gpu_cycle = cycle(args.transfer_gpus)
            transfer_args = SimpleNamespace(
                source_dataset=args.dataset,
                pretrained_name=model_name,
                config=args.config,
                para_config=args.para_config,
            )

            for target in target_datasets:
                gpu_id = next(gpu_cycle)
                print(
                    f"[CURVATURE] >> Repeat {repeat_idx}/{args.repeats}, transferring to {target} on GPU {gpu_id}..."
                )
                result = run_transfer(target, gpu_id, transfer_args, curvature_dir, repeat_idx)
                if result["target_acc"] is None:
                    raise RuntimeError(
                        f"Could not parse target accuracy for dataset '{target}' (curvature {curvature})."
                    )
                aggregated[target].append(result["target_acc"])
                result.update({
                    "curvature": curvature,
                    "model_name": model_name,
                })
                run_records.append(result)

        formatted = {}
        for column, values in aggregated.items():
            if not values:
                continue
            mean_val = np_mean(values)
            std_val = np_std(values, ddof=1) if len(values) > 1 else 0.0
            formatted[column] = f"{mean_val:.4f}±{std_val:.4f}"
        summary_rows[f"curv_{curvature:.4g}"] = formatted

    ordered_rows = []
    ordered_index = []
    for curvature in curvatures:
        label = f"curv_{curvature:.4g}"
        row = summary_rows.get(label, {})
        ordered_rows.append({col: row.get(col, "") for col in column_order})
        ordered_index.append(label)

    df = pd.DataFrame(ordered_rows, index=ordered_index)
    csv_path = exp_root / "curvature_matrix.csv"
    df.to_csv(csv_path)

    runs_path = exp_root / "runs.jsonl"
    with runs_path.open("w", encoding="utf-8") as f:
        for record in run_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "targets": target_datasets,
        "curvatures": curvatures,
        "repeats": args.repeats,
        "pretext": args.pretext,
        "pretrain_gpu": args.pretrain_gpu,
        "transfer_gpus": args.transfer_gpus,
        "config": args.config,
        "para_config": args.para_config,
        "direct_device": args.direct_device,
        "direct_epochs": args.direct_epochs,
        "is_reduction": args.is_reduction,
        "column_order": column_order,
        "command": "python " + " ".join(os.sys.argv),
        "outputs": {
            "summary_csv": str(csv_path),
            "runs_log": str(runs_path),
        },
    }
    (exp_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[CURVATURE] Aggregated curvature sweep saved to: {csv_path}")


if __name__ == "__main__":
    main()

