#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse experiment_log_*.txt files, reconstruct per-experiment CSVs,
and produce master & summary tables.

Usage:
  python parse_logs_to_csv.py --log_dir . --out_dir ./parsed_csv

Notes:
- This script expects log lines like:
    "=== Best @ epoch 144: val=0.9203, test=0.9261 ==="
    "Experiment Photo-/mnt/.../Computers....pth-False-10-5 completed in 107.87 seconds."
- It returns TEST accuracy as the metric for scientific reporting.
"""

import os
import re
import csv
import argparse
from glob import glob
from collections import defaultdict, namedtuple
from statistics import mean, pstdev

BEST_RE = re.compile(
    r"Best\s*@\s*epoch\s*(?P<epoch>\d+):\s*val=(?P<val>[0-9.]+),\s*test=(?P<test>[0-9.]+)",
    flags=re.IGNORECASE
)
# Experiment ID format from your runner:
# {test_dataset}-{pretrained_path}-{few}-{shot}-{run_idx}
COMPLETED_RE = re.compile(
    r"Experiment\s+(?P<expid>.+?)\s+completed\s+in\s+(?P<secs>[0-9.]+)\s+seconds\.",
    flags=re.IGNORECASE
)
EXP_ID_RE = re.compile(
    r"^(?P<test_dataset>[^-]+)-(?P<pretrained>.+)-(?P<few>True|False)-(?P<shot>\d+)-(?P<run>\d+)$"
)

RunRec = namedtuple("RunRec",
                    ["test_dataset", "pretrained", "few", "shot", "run",
                     "epoch", "val_acc", "test_acc", "duration_sec", "log_file"])

def abbrev_from_ckpt_first_token(pretrained_path: str) -> str:
    """
    根据 pth 文件名的第一段生成 HGL-** 别名：
    例：PubMed.* -> HGL-PM, CiteSeer.* -> HGL-CS, 其余按首字母缩写。
    """
    base = os.path.basename(pretrained_path)
    first = base.split(".")[0] if "." in base else base
    # 简单缩写策略：抓取大写字母；若没有，就取前两位
    uppers = "".join([c for c in first if c.isupper()])
    if first.lower().startswith("pubmed"):
        tag = "PM"
    elif first.lower().startswith("citeseer"):
        tag = "CS"
    else:
        tag = uppers if uppers else first[:2].title()
    return f"HGL-{tag}"

def fewshot_label(few: str, shot: int) -> str:
    if str(few) == "False":
        return "Public"
    if shot == 5:
        return "5shot"
    if shot == 10:
        return "10shot"
    return f"{shot}shot"

def parse_logs(log_paths):
    """
    扫描所有日志文件，匹配 'Best ... test=...' 以及 'Experiment ... completed in ...',
    将最近一次 Best 与随后出现的该 Experiment 完成行配对。
    """
    run_records = []
    for lp in sorted(log_paths):
        try:
            with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                last_best = None  # (epoch, val, test)
                for line in f:
                    m_best = BEST_RE.search(line)
                    if m_best:
                        last_best = (
                            int(m_best.group("epoch")),
                            float(m_best.group("val")),
                            float(m_best.group("test")),
                        )
                        continue

                    m_done = COMPLETED_RE.search(line)
                    if m_done:
                        expid = m_done.group("expid").strip()
                        secs = float(m_done.group("secs"))
                        m_id = EXP_ID_RE.match(expid)
                        if not m_id:
                            # 无法解析 expid 就跳过该条
                            continue
                        if last_best is None:
                            # 找不到对应的 Best 行，也生成记录但 test_acc 为 None
                            epoch, val, test = (None, None, None)
                        else:
                            epoch, val, test = last_best

                        run_records.append(RunRec(
                            test_dataset=m_id.group("test_dataset"),
                            pretrained=m_id.group("pretrained"),
                            few=m_id.group("few"),
                            shot=int(m_id.group("shot")),
                            run=int(m_id.group("run")),
                            epoch=epoch, val_acc=val, test_acc=test,
                            duration_sec=secs, log_file=os.path.basename(lp)
                        ))
                        # 如果一个日志里连续多个实验，last_best 可能继续被后续实验复用；
                        # 不清空以容忍 “Best 在前，Completed 在后” 的多段结构。
        except Exception as e:
            print(f"[WARN] Failed to parse {lp}: {e}")
    return run_records

def write_master_csv(out_dir, run_records):
    os.makedirs(out_dir, exist_ok=True)
    master_csv = os.path.join(out_dir, "master_runs.csv")
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["test_dataset", "pretrained", "few", "shot", "run",
                    "epoch", "val_acc", "test_acc", "duration_sec", "log_file"])
        for r in sorted(run_records, key=lambda x: (x.test_dataset, x.pretrained, x.few, x.shot, x.run)):
            w.writerow([r.test_dataset, r.pretrained, r.few, r.shot, r.run,
                        r.epoch if r.epoch is not None else "",
                        f"{r.val_acc:.4f}" if r.val_acc is not None else "",
                        f"{r.test_acc:.4f}" if r.test_acc is not None else "",
                        f"{r.duration_sec:.2f}" if r.duration_sec is not None else "",
                        r.log_file])
    print(f"[OK] Wrote {master_csv}")
    return master_csv

def write_per_combo_csvs(out_dir, run_records):
    """
    每个 (test_dataset, pretrained, few, shot) 组合一个 CSV，记录 1..n 次 run 的 test_acc + duration
    """
    combos = defaultdict(list)
    for r in run_records:
        key = (r.test_dataset, r.pretrained, r.few, r.shot)
        combos[key].append(r)

    combo_dir = os.path.join(out_dir, "per_experiment")
    os.makedirs(combo_dir, exist_ok=True)

    for key, rs in combos.items():
        test_dataset, pretrained, few, shot = key
        base = os.path.basename(pretrained)
        fname = f"{test_dataset}__{base}__{fewshot_label(few, shot)}.csv"
        path = os.path.join(combo_dir, fname)

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "test_acc", "val_acc", "epoch", "duration_sec", "log_file"])
            for r in sorted(rs, key=lambda x: x.run):
                w.writerow([
                    r.run,
                    f"{r.test_acc:.4f}" if r.test_acc is not None else "",
                    f"{r.val_acc:.4f}" if r.val_acc is not None else "",
                    r.epoch if r.epoch is not None else "",
                    f"{r.duration_sec:.2f}" if r.duration_sec is not None else "",
                    r.log_file
                ])
        print(f"[OK] Wrote {path}")

def write_summary_tables(out_dir, run_records):
    """
    写摘要表：每个组合的均值/标准差/完成次数；以及一个透视表用于做论文中的大表
    """
    os.makedirs(out_dir, exist_ok=True)
    combos = defaultdict(list)
    for r in run_records:
        key = (r.test_dataset, r.pretrained, r.few, r.shot)
        if r.test_acc is not None:
            combos[key].append(r.test_acc)

    # 汇总表
    sum_csv = os.path.join(out_dir, "summary_by_combo.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["HGL_alias", "test_dataset", "few_label",
                    "pretrained_basename", "few", "shot",
                    "n_runs", "mean_test_acc", "std_test_acc"])
        for (test_dataset, pretrained, few, shot), vals in sorted(combos.items()):
            n = len(vals)
            m = mean(vals) if n else None
            s = pstdev(vals) if n > 1 else 0.0 if n == 1 else None
            alias = abbrev_from_ckpt_first_token(pretrained)
            w.writerow([
                alias, test_dataset, fewshot_label(few, shot),
                os.path.basename(pretrained), few, shot, n,
                f"{m:.4f}" if m is not None else "",
                f"{s:.4f}" if s is not None else ""
            ])
    print(f"[OK] Wrote {sum_csv}")

    # 透视表：行=HGL别名，列=测试集 × (Public/5shot/10shot)，值=mean±std
    # 先把均值/方差放入字典
    stats = defaultdict(dict)  # key1=alias, key2=(dataset, label) -> "mean±std"
    for (test_dataset, pretrained, few, shot), vals in combos.items():
        alias = abbrev_from_ckpt_first_token(pretrained)
        label = fewshot_label(few, shot)
        n = len(vals)
        if n == 0:
            cell = ""
        elif n == 1:
            cell = f"{mean(vals):.4f}±0.0000"
        else:
            cell = f"{mean(vals):.4f}±{pstdev(vals):.4f}"
        stats[alias][(test_dataset, label)] = cell

    # 收集所有列键
    datasets = sorted({k[0] for d in stats.values() for k in d.keys()})
    labels = ["Public", "5shot", "10shot"]
    headers = ["HGL_alias"] + [f"{ds}__{lb}" for ds in datasets for lb in labels]
    pivot_csv = os.path.join(out_dir, "pivot_table.csv")
    with open(pivot_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for alias in sorted(stats.keys()):
            row = [alias]
            for ds in datasets:
                for lb in labels:
                    row.append(stats[alias].get((ds, lb), ""))
            w.writerow(row)
    print(f"[OK] Wrote {pivot_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default=".", help="Directory containing experiment_log_*.txt")
    ap.add_argument("--out_dir", type=str, default="./parsed_csv", help="Where to write CSV outputs")
    args = ap.parse_args()

    log_glob = os.path.join(args.log_dir, "experiment_log_*.txt")
    paths = glob(log_glob)
    if not paths:
        print(f"[ERR] No logs matched: {log_glob}")
        return

    runs = parse_logs(paths)
    if not runs:
        print("[ERR] No run records parsed from logs.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    master = write_master_csv(args.out_dir, runs)
    write_per_combo_csvs(args.out_dir, runs)
    write_summary_tables(args.out_dir, runs)

    print("\n[INFO] Done. Key outputs:")
    print(f"- Master runs: {master}")
    print(f"- Per-experiment CSVs: {os.path.join(args.out_dir, 'per_experiment')}")
    print(f"- Summary by combo: {os.path.join(args.out_dir, 'summary_by_combo.csv')}")
    print(f"- Pivot table: {os.path.join(args.out_dir, 'pivot_table.csv')}")

if __name__ == "__main__":
    main()
