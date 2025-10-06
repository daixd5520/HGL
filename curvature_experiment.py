#!/usr/bin/env python3
"""End-to-end curvature sweep automation for HypGraphLoRA.

This utility schedules pre-training and transfer runs for multiple initial
curvature values, dispatching each job to one GPU (using CUDA_VISIBLE_DEVICES)
and collating the averaged accuracy (mean ± std, in percentage) into a CSV.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, List, Optional, Sequence, TextIO, Tuple

import yaml

BEST_LINE_RE = re.compile(
    r"Best\s*@\s*epoch\s*\d+\s*:\s*val\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*test\s*=\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_DIR = os.path.join(REPO_ROOT, "pre_trained_gnn")


def parse_float_list(values: Sequence[str]) -> List[float]:
    result: List[float] = []
    for item in values:
        for token in str(item).replace(';', ',').split(','):
            token = token.strip()
            if not token:
                continue
            result.append(float(token))
    return result


def parse_str_list(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for item in values:
        for token in str(item).replace(';', ',').split(','):
            token = token.strip()
            if token:
                result.append(token)
    return result


def parse_int_list(values: Sequence[str]) -> List[int]:
    return [int(v) for v in parse_str_list(values)]


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"true", "t", "yes", "y", "1"}:
        return True
    if value in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as boolean")


def curvature_token(value: float) -> str:
    fixed = f"{value:.4f}"
    sanitized = fixed.replace('-', 'm').replace('.', 'p')
    return f"curv_{sanitized}"


def build_model_filename(
    curvature: float,
    pretrain_dataset: str,
    pretext: str,
    gnn_type: str,
    hyperbolic: bool,
    is_reduction: bool,
    *,
    reuse_existing: bool,
) -> Tuple[str, str]:
    """Create a descriptive checkpoint filename, optionally avoiding overwrites."""

    base_name = ".".join(
        [
            pretrain_dataset,
            pretext,
            gnn_type,
            curvature_token(curvature),
            f"hyp_{hyperbolic}",
            str(is_reduction),
        ]
    ) + ".pth"

    base_path = os.path.join(PRETRAIN_DIR, base_name)
    if reuse_existing:
        return base_name, base_path

    if os.path.exists(base_path):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        stamped_name = base_name[:-4] + f".{stamp}.pth"
        return stamped_name, os.path.join(PRETRAIN_DIR, stamped_name)

    return base_name, base_path


class GPUJob:
    """A single command scheduled onto one GPU."""

    def __init__(
        self,
        description: str,
        command: Sequence[str],
        log_path: str,
        on_complete: Optional[Callable[["GPUJob", int], None]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.description = description
        self.command = list(command)
        self.log_path = log_path
        self.on_complete = on_complete
        self.env = env or {}

    def spawn(self, gpu_id: int) -> Tuple[subprocess.Popen, TextIO]:
        env = os.environ.copy()
        env.update(self.env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("PYTHONUNBUFFERED", "1")

        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_handle = open(self.log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            self.command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return proc, log_handle


def scheduler(queue: Deque[GPUJob], gpus: Sequence[int]) -> None:
    running: Dict[int, Tuple[subprocess.Popen, GPUJob, TextIO]] = {}
    try:
        while queue or running:
            # Launch new jobs when GPUs are free.
            for gpu_id in gpus:
                if gpu_id in running:
                    continue
                if not queue:
                    continue
                job = queue.popleft()
                proc, log_handle = job.spawn(gpu_id)
                running[gpu_id] = (proc, job, log_handle)
                print(f"[GPU{gpu_id}] Launch {job.description}")

            time.sleep(1.0)

            # Check running jobs.
            for gpu_id in list(running.keys()):
                proc, job, log_handle = running[gpu_id]
                ret = proc.poll()
                if ret is None:
                    continue
                log_handle.close()
                running.pop(gpu_id, None)
                print(f"[GPU{gpu_id}] Finish {job.description} (exit={ret})")
                if ret != 0:
                    raise RuntimeError(
                        f"Job failed with exit code {ret}: {job.description}. Log: {job.log_path}"
                    )
                if job.on_complete:
                    job.on_complete(job, ret)
    except KeyboardInterrupt:
        print("Received Ctrl+C, terminating running jobs...")
        for gpu_id, (proc, job, _) in running.items():
            proc.terminate()
            print(f"[GPU{gpu_id}] Terminated {job.description}")
        raise
    finally:
        for gpu_id, (proc, job, log_handle) in list(running.items()):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            log_handle.close()


def extract_best_test(log_path: str) -> float:
    with open(log_path, "r", encoding="utf-8") as f:
        test_acc = None
        for line in f:
            match = BEST_LINE_RE.search(line)
            if match:
                test_acc = float(match.group(2))
        if test_acc is None:
            raise ValueError(f"Could not parse test accuracy from {log_path}")
        return test_acc


def enqueue_transfer_jobs(
    curvature: float,
    model_filename: str,
    args: argparse.Namespace,
    queue: Deque[GPUJob],
    test_datasets: Sequence[str],
    logs_root: str,
    results: Dict[Tuple[float, str], List[float]],
) -> int:
    """Schedule (or reuse) transfer jobs for a curvature checkpoint."""

    scheduled = 0
    for dataset in test_datasets:
        for run_idx in range(1, args.runs + 1):
            dataset_dir = os.path.join(logs_root, "transfer", dataset)
            log_path = os.path.join(
                dataset_dir,
                f"{curvature_token(curvature)}_run{run_idx}.log",
            )

            if args.reuse_logs and os.path.exists(log_path):
                try:
                    acc = extract_best_test(log_path)
                except Exception as exc:  # pragma: no cover - defensive
                    print(
                        f"[Transfer] Existing log for {dataset} (c={curvature:.4f}, run={run_idx}) "
                        f"could not be parsed ({exc}); scheduling rerun."
                    )
                else:
                    results[(curvature, dataset)].append(acc)
                    print(
                        f"[Transfer] Reused {dataset} c={curvature:.4f} run={run_idx} -> test={acc:.4f}"
                    )
                    continue

            transfer_job = build_transfer_job(
                curvature,
                dataset,
                run_idx,
                model_filename,
                args,
                logs_root,
                results,
                log_path,
            )
            queue.append(transfer_job)
            scheduled += 1
    return scheduled


def build_pretrain_job(
    curvature: float,
    model_filename: str,
    args: argparse.Namespace,
    queue: Deque[GPUJob],
    test_datasets: Sequence[str],
    logs_root: str,
    results: Dict[Tuple[float, str], List[float]],
) -> GPUJob:
    python_bin = sys.executable
    log_path = os.path.join(logs_root, "pretrain", f"{curvature_token(curvature)}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    command = [
        python_bin,
        "main.py",
        "--is_pretrain",
        "True",
        "--is_transfer",
        "False",
        "--pretrain_dataset",
        args.pretrain_dataset,
        "--pretext",
        args.pretext,
        "--config",
        args.config,
        "--para_config",
        args.para_config,
        "--gpu_id",
        "0",
        "--pretrain_curvature",
        f"{curvature}",
        "--pretrain_output_name",
        model_filename,
        "--is_reduction",
        "True" if args.is_reduction else "False",
    ]

    def _on_complete(job: GPUJob, _ret: int) -> None:
        model_path = os.path.join(PRETRAIN_DIR, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Expected checkpoint not found: {model_path}")
        print(f"[Pretrain] Ready checkpoint {model_filename}")
        scheduled = enqueue_transfer_jobs(
            curvature,
            model_filename,
            args,
            queue,
            test_datasets,
            logs_root,
            results,
        )
        if scheduled:
            print(
                f"[Pretrain] Enqueued {scheduled} transfer jobs for curvature {curvature:.4f}"
            )

    return GPUJob(
        description=f"pretrain(c={curvature:.4f})",
        command=command,
        log_path=log_path,
        on_complete=_on_complete,
    )


def build_transfer_job(
    curvature: float,
    dataset: str,
    run_idx: int,
    model_filename: str,
    args: argparse.Namespace,
    logs_root: str,
    results: Dict[Tuple[float, str], List[float]],
    log_path: str,
) -> GPUJob:
    python_bin = sys.executable
    dataset_dir = os.path.dirname(log_path)
    os.makedirs(dataset_dir, exist_ok=True)

    command = [
        python_bin,
        "main.py",
        "--is_pretrain",
        "False",
        "--is_transfer",
        "True",
        "--pretrain_dataset",
        args.pretrain_dataset,
        "--test_dataset",
        dataset,
        "--config",
        args.config,
        "--para_config",
        args.para_config,
        "--gpu_id",
        "0",
        "--pretrained_model_name",
        model_filename,
        "--curvature",
        f"{curvature}",
        "--is_reduction",
        "True" if args.is_reduction else "False",
    ]

    def _on_complete(job: GPUJob, _ret: int) -> None:
        acc = extract_best_test(job.log_path)
        results[(curvature, dataset)].append(acc)
        print(
            f"[Transfer] {dataset} c={curvature:.4f} run={run_idx} -> test={acc:.4f}"
        )

    return GPUJob(
        description=f"transfer({dataset}, c={curvature:.4f}, run={run_idx})",
        command=command,
        log_path=log_path,
        on_complete=_on_complete,
    )


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curvature hyperparameter sweep automation. Suggested range: 0.1~2.0."
    )
    parser.add_argument(
        "--curvatures",
        nargs="+",
        required=True,
        help="List (or comma-separated string) of initial curvature values to evaluate.",
    )
    parser.add_argument("--pretrain-dataset", dest="pretrain_dataset", required=True)
    parser.add_argument(
        "--test-datasets",
        nargs="+",
        required=True,
        help="One or more target datasets for transfer evaluation.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=[str(i) for i in range(8)],
        help="GPU ids available for scheduling (default: 0-7).",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of transfer runs per curvature.")
    parser.add_argument("--pretext", type=str, default="GRACE")
    parser.add_argument("--config", type=str, default="./config.yaml")
    parser.add_argument("--para_config", type=str, default="./config2.yaml")
    parser.add_argument(
        "--is-reduction",
        dest="is_reduction",
        type=str2bool,
        default=False,
        help="Whether to enable feature reduction during pretraining/transfer.",
    )
    parser.add_argument(
        "--reuse-checkpoints",
        dest="reuse_checkpoints",
        type=str2bool,
        default=False,
        help="Skip pretraining when a matching checkpoint already exists.",
    )
    parser.add_argument(
        "--reuse-logs",
        dest="reuse_logs",
        type=str2bool,
        default=False,
        help="Reuse existing transfer logs when available to avoid reruns.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Destination CSV file (default: result/curvature_<timestamp>.csv)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional folder name under experiments/ to store logs.",
    )

    args = parser.parse_args()

    curvatures = parse_float_list(args.curvatures)
    if not curvatures:
        raise ValueError("No valid curvature values provided")
    test_datasets = parse_str_list(args.test_datasets)
    if not test_datasets:
        raise ValueError("No valid test datasets provided")
    gpus = parse_int_list(args.gpus)
    if not gpus:
        raise ValueError("At least one GPU id must be provided")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    with open(os.path.join(REPO_ROOT, args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.pretrain_dataset not in config:
        raise KeyError(f"Pretrain dataset '{args.pretrain_dataset}' not found in {args.config}")
    pre_cfg = config[args.pretrain_dataset]
    gnn_type = pre_cfg.get("gnn_type", "GAT")
    hyperbolic = bool(pre_cfg.get("hyperbolic_backbone", True))

    exp_timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = args.experiment_name or f"curvature_{args.pretrain_dataset}_{exp_timestamp}"
    logs_root = ensure_dir(os.path.join(REPO_ROOT, "experiments", exp_name))
    ensure_dir(os.path.join(logs_root, "pretrain"))
    for dataset in test_datasets:
        ensure_dir(os.path.join(logs_root, "transfer", dataset))

    queue: Deque[GPUJob] = deque()
    results: Dict[Tuple[float, str], List[float]] = defaultdict(list)

    for curvature in curvatures:
        model_filename, model_path = build_model_filename(
            curvature,
            args.pretrain_dataset,
            args.pretext,
            gnn_type,
            hyperbolic,
            args.is_reduction,
            reuse_existing=args.reuse_checkpoints,
        )

        if args.reuse_checkpoints and os.path.exists(model_path):
            print(
                f"[Pretrain] Reusing existing checkpoint for curvature {curvature:.4f}: {model_filename}"
            )
            enqueue_transfer_jobs(
                curvature,
                model_filename,
                args,
                queue,
                test_datasets,
                logs_root,
                results,
            )
            continue

        pretrain_job = build_pretrain_job(
            curvature,
            model_filename,
            args,
            queue,
            test_datasets,
            logs_root,
            results,
        )
        queue.append(pretrain_job)

    total_expected = args.runs * len(curvatures) * len(test_datasets)
    print(
        f"Scheduled {len([j for j in queue if 'pretrain' in j.description])} pretrains."
        f" Expecting up to {total_expected} transfer runs (reused logs reduce this)."
    )
    scheduler(queue, gpus)

    # Aggregate results.
    rows: List[List[str]] = []
    for curvature in curvatures:
        for dataset in test_datasets:
            key = (curvature, dataset)
            values = results.get(key, [])
            if len(values) != args.runs:
                raise RuntimeError(
                    f"Expected {args.runs} runs for curvature {curvature} on {dataset}, got {len(values)}"
                )
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            percent_str = f"{mean * 100:.2f}±{std * 100:.2f}"
            rows.append(
                [
                    args.pretrain_dataset,
                    dataset,
                    f"{curvature:.4f}",
                    percent_str,
                ]
            )

    rows.sort(key=lambda r: (float(r[2]), r[1]))

    output_csv = args.output_csv or os.path.join(
        REPO_ROOT,
        "result",
        f"curvature_{args.pretrain_dataset}_{exp_timestamp}.csv",
    )
    ensure_dir(os.path.dirname(output_csv))
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pretrain_dataset", "test_dataset", "curvature", "mean±std (%)"])
        writer.writerows(rows)

    print(f"Saved summary to {output_csv}")


if __name__ == "__main__":
    main()
