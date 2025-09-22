# auto.py  /  exp_multicard.py
import argparse
import os
import re
import time
import json
import signal
import subprocess
from datetime import datetime
from itertools import product
from multiprocessing import Process, JoinableQueue, Queue, Event

import pandas as pd


# ========================= 配置 =========================
FEW_SETTINGS = [
    {"few": False, "shot": 0,  "label": "Public"},
    {"few": True,  "shot": 5,  "label": "5shot"},
    {"few": True,  "shot": 10, "label": "10shot"},
]

# main.py 命令模板（你的 main 是单卡，不必传 --gpu_id；如果需要可在末尾加上 " --gpu_id 0"）
CMD_TEMPLATE = (
    "python main.py --is_transfer True "
    "--test_dataset {test_dataset} "
    "--pretrained_model_name {pretrained_model_path} "
    "{few_part} {shot_part}"
)

# 匹配：=== Best @ epoch 78: val=0.8494, test=0.8329 ===
BEST_LINE_RE = re.compile(
    r"Best\s*@\s*epoch\s*\d+\s*:\s*val\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*test\s*=\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

DATASET_TO_ABBR = {"Cora":"C","CiteSeer":"CS","PubMed":"PM","Photo":"P","Computers":"Com"}


# ========================= 工具函数 =========================
def now_str(): return datetime.now().strftime("%Y%m%d-%H%M%S")
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def src_from_model(p): return os.path.basename(p).split(".")[0]
def row_label_from_model(p):
    src = src_from_model(p)
    return f"HGL-{DATASET_TO_ABBR.get(src, src[:2].upper())}"


# ========================= 核心执行（单次 run） =========================
def run_once(task, log_root, stop_event: Event):
    """
    在指定 GPU 上跑一次实验；写 stdout/stderr/meta；返回结果 dict。
    使用 CUDA_VISIBLE_DEVICES=<gpu_id> 隔离。支持在 stop_event 置位后优雅退出/强杀。
    """
    test_dataset = task["test_dataset"]
    pretrained_model_path = task["pretrained_model_path"]
    few = task["few"]; shot = task["shot"]; run_idx = task["run_idx"]; gpu_id = task["gpu_id"]

    few_part = "--few True" if few else "--few False"
    shot_part = f"--shot {shot}" if few else ""
    cmd = CMD_TEMPLATE.format(
        test_dataset=test_dataset,
        pretrained_model_path=pretrained_model_path,
        few_part=few_part, shot_part=shot_part
    )

    src_label = row_label_from_model(pretrained_model_path)
    exp_tag = f"{src_label}__test-{test_dataset}__{('few' if few else 'public')}{(f'-{shot}shot' if few else '')}__run{run_idx}"
    run_dir = ensure_dir(os.path.join(log_root, src_label, test_dataset))
    stdout_path = os.path.join(run_dir, f"{exp_tag}.stdout.txt")
    stderr_path = os.path.join(run_dir, f"{exp_tag}.stderr.txt")
    meta_path   = os.path.join(run_dir, f"{exp_tag}.meta.json")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 关键：让子进程只“看见”这一块 GPU

    # 如果 stop 已触发，直接返回空结果
    if stop_event.is_set():
        return _empty_result(pretrained_model_path, test_dataset, few, shot, run_idx, gpu_id)

    print(f"[GPU{gpu_id}] START {exp_tag}")
    t0 = time.time()
    proc = None
    out, err = "", ""
    try:
        # 独立进程组，便于 killpg
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, universal_newlines=True, bufsize=1, start_new_session=True
        )
        out, err = proc.communicate()
    except Exception as e:
        err = (err or "") + f"\n[auto.py] Exception while running: {e}\n"
    finally:
        dura = time.time() - t0

        # 若收到停机信号而子进程仍在，尝试终止
        if stop_event.is_set() and proc and proc.poll() is None:
            _terminate_proc_group(proc)

        # 写日志文件
        with open(stdout_path, "w", encoding="utf-8") as f: f.write(out or "")
        with open(stderr_path, "w", encoding="utf-8") as f: f.write(err or "")

        # 解析 test/val
        test_acc, val_acc = None, None
        for line in (out or "").splitlines():
            m = BEST_LINE_RE.search(line)
            if m:
                try:
                    val_acc = float(m.group(1))
                    test_acc = float(m.group(2))
                except Exception:
                    pass

        meta = {
            "cmd": cmd, "gpu_id": gpu_id, "few": few, "shot": shot,
            "test_dataset": test_dataset, "pretrained_model_path": pretrained_model_path,
            "exit_code": (proc.returncode if proc else None), "duration_sec": dura,
            "parsed_val_acc": val_acc, "parsed_test_acc": test_acc,
            "stdout_path": stdout_path, "stderr_path": stderr_path,
            "time": datetime.now().isoformat(timespec="seconds"),
            "stopped": stop_event.is_set(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if test_acc is None:
            print(f"[GPU{gpu_id}] DONE {exp_tag} (no test_acc found, dura={dura:.1f}s)")
            test_acc = 0.0
        else:
            print(f"[GPU{gpu_id}] DONE {exp_tag} (test={test_acc:.4f}, dura={dura:.1f}s)")

        return {
            "row_label": row_label_from_model(pretrained_model_path),
            "src_dataset": src_from_model(pretrained_model_path),
            "test_dataset": test_dataset,
            "few_label": "Public" if not few else f"{shot}shot",
            "run_idx": run_idx,
            "test_acc": test_acc,
            "val_acc": val_acc,
            "duration_sec": dura,
            "gpu_id": gpu_id,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "meta_path": meta_path,
        }


def _terminate_proc_group(proc: subprocess.Popen):
    """温柔 SIGTERM，超时再 SIGKILL"""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def _empty_result(pretrained_model_path, test_dataset, few, shot, run_idx, gpu_id):
    return {
        "row_label": row_label_from_model(pretrained_model_path),
        "src_dataset": src_from_model(pretrained_model_path),
        "test_dataset": test_dataset,
        "few_label": "Public" if not few else f"{shot}shot",
        "run_idx": run_idx,
        "test_acc": 0.0, "val_acc": None, "duration_sec": 0.0,
        "gpu_id": gpu_id, "stdout_path": "", "stderr_path": "", "meta_path": "",
    }


# ========================= GPU Worker（每卡一个） =========================
def gpu_worker(gpu_id: int, task_queue: JoinableQueue, result_queue: Queue, log_root: str, stop_event: Event):
    while True:
        task = task_queue.get()
        try:
            if task is None:
                # 哨兵：标记完成并退出
                return
            if stop_event.is_set():
                # 停机后不再消费，直接返回占位结果
                task["gpu_id"] = gpu_id
                result_queue.put(_empty_result(
                    task["pretrained_model_path"], task["test_dataset"],
                    task["few"], task["shot"], task["run_idx"], gpu_id
                ))
                continue
            task["gpu_id"] = gpu_id
            res = run_once(task, log_root, stop_event)
            result_queue.put(res)
        except Exception as e:
            # 发生异常也塞一条占位结果，避免主进程阻塞
            result_queue.put({
                "row_label": row_label_from_model(task.get("pretrained_model_path","")),
                "src_dataset": src_from_model(task.get("pretrained_model_path","unknown.pth")),
                "test_dataset": task.get("test_dataset",""),
                "few_label": "Public" if not task.get("few") else f"{task.get('shot') }shot",
                "run_idx": task.get("run_idx", -1),
                "test_acc": 0.0, "val_acc": None, "duration_sec": 0.0,
                "gpu_id": gpu_id, "stdout_path": "", "stderr_path": "", "meta_path": "",
                "error": str(e),
            })
        finally:
            # JoinableQueue 需要 task_done
            task_queue.task_done()


# ========================= 汇总出表 =========================
def aggregate(all_results, out_root):
    ensure_dir(out_root)
    df = pd.DataFrame(all_results)
    long_csv = os.path.join(out_root, "runs_long.csv")
    df.to_csv(long_csv, index=False)

    stat = (
        df.groupby(["row_label","test_dataset","few_label"])["test_acc"]
          .agg(mean="mean", std="std").reset_index()
    )
    stat["mean_std"] = stat.apply(lambda x: f"{x['mean']:.4f} ± {x['std']:.4f}", axis=1)
    wide = stat.pivot(index="row_label", columns=["test_dataset","few_label"], values="mean_std")
    wide_csv = os.path.join(out_root, "table_wide.csv")
    wide.to_csv(wide_csv)

    print(f"[SAVE] 长表: {long_csv}")
    print(f"[SAVE] 宽表: {wide_csv}")


# ========================= 主程序 =========================
def main():
    ap = argparse.ArgumentParser("GPU-scheduled GraphLoRA auto runner (1 GPU = 1 process)")
    ap.add_argument("--models", nargs="+", required=True, help="一个或多个 .pth 预训练模型完整路径")
    ap.add_argument("--tests",  nargs="+", required=True, help="一个或多个测试数据集名字（如 Photo Cora PubMed ...）")
    ap.add_argument("--runs", type=int, default=5, help="每种设置重复次数（默认5）")
    ap.add_argument("--gpus", type=int, default=5, help="可用 GPU 数量（默认5，对应 0..gpus-1）")
    ap.add_argument("--out",  type=str, default=f"./experiments/{now_str()}", help="输出根目录")
    args = ap.parse_args()

    out_root = ensure_dir(args.out)
    log_root = ensure_dir(os.path.join(out_root, "logs"))

    # 生成任务
    tasks = []
    for model_path, test_dataset in product(args.models, args.tests):
        for s in FEW_SETTINGS:
            for r in range(1, args.runs+1):
                tasks.append({
                    "pretrained_model_path": model_path,
                    "test_dataset": test_dataset,
                    "few": s["few"], "shot": s["shot"], "run_idx": r
                })

    total = len(tasks)
    print(f"[PLAN] models={len(args.models)}, tests={len(args.tests)}, few={len(FEW_SETTINGS)}, runs={args.runs} → total tasks={total}")
    print(f"[OUT] {out_root}")

    # 停机事件 & 信号处理
    stop_event = Event()
    def _signal_handler(signum, frame):
        print(f"[SIGNAL] Caught {signum}. Stopping all workers …")
        stop_event.set()
        # 给每个队列补哨兵，防止 worker 阻塞
        for q in task_queues:
            try: q.put_nowait(None)
            except Exception: pass
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 建立每 GPU 一个 JoinableQueue + worker（非 daemon，便于优雅退出）
    global task_queues  # 供信号处理器访问
    task_queues = [JoinableQueue() for _ in range(args.gpus)]
    result_queue = Queue()
    workers = []
    for gid in range(args.gpus):
        p = Process(target=gpu_worker, args=(gid, task_queues[gid], result_queue, log_root, stop_event))
        p.start()
        workers.append(p)

    # 轮询分发任务
    for i, t in enumerate(tasks):
        task_queues[i % args.gpus].put(t)
    # 正常结束哨兵
    for q in task_queues:
        q.put(None)

    # 等所有队列消费完
    for q in task_queues:
        q.join()

    # 主进程收集结果
    all_results = []
    done = 0
    while done < total:
        res = result_queue.get()
        all_results.append(res)
        done += 1
        print(f"[PROGRESS] {done}/{total} finished.")

    # 等 worker 退出；若迟迟不退则 terminate
    for p in workers:
        p.join(timeout=15)
    for p in workers:
        if p.is_alive():
            print(f"[WARN] Worker {p.pid} still alive, terminating …")
            p.terminate()

    # 汇总表与追溯信息
    aggregate(all_results, out_root)
    with open(os.path.join(out_root, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_root, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for t in tasks: f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
