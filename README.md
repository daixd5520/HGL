# HypGraphLoRA

## Requirements
```
python==3.11.5
torch==2.1.0
cuda==12.1
numpy==1.26.0
torch_geometric==2.4.0
```

## How to Run
You can easily run our code by

```
# Pre-training
 python main.py --is_pretrain True --is_transfer False --pretrain_dataset CiteSeer

# Fine-tuning
python main.py --is_transfer True --pretrained_model_name /mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/PubMed.GRACE.GAT.hyp_True.True.20250911-161024.pth --lora_alpha 32 --r 16

# ----------------------这个能够生成表格
python exp_multicard.py \
--models /mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/PubMed.GRACE.GAT.hyp_True.True.20250911-161024.pth \
--tests PubMed Cora Photo \
--runs 5 \
--gpus 5

```

## Forgetting Experiment Runner

The multi-GPU forgetting experiment script orchestrates direct evaluation on the
pre-training dataset (d1) and fine-tuning on up to four additional datasets (d2)
in parallel—one per GPU. It aggregates all results into a CSV matrix for easy
comparison.

### 1. Prepare your environment

- Make sure the pretrained checkpoint you want to evaluate has already been
  produced. The script infers the source dataset from the checkpoint filename,
  so the file must start with the dataset name (e.g., `Cora.GRACE.GAT....pth`).
- Place the checkpoint anywhere that the current user can read. The script will
  copy it into `pre_trained_gnn/` automatically if it is not already there.
- Ensure all target datasets are available locally under `./datasets/` or can
  be downloaded automatically by the existing dataset loaders.

### 2. Run the experiment

```bash
python experiments/forgetting_experiment.py \
  --pretrained-model /path/to/Cora.GRACE.GAT.hyp_True.True.20250911-161024.pth \
  --d2 CiteSeer PubMed Photo Computers \
  --gpus 0 1 2 3
```

Key flags:

- `--pretrained-model`: Absolute or relative path to the pretrained `.pth`
  checkpoint. Required.
- `--d2`: Optional list of target datasets to fine-tune on. If omitted, the
  script runs on every supported dataset except the inferred source dataset.
- `--gpus`: GPU indices to use. The script assigns one dataset per GPU in the
  provided order. Provide four IDs (e.g., `0 1 2 3`) to run four transfers in
  parallel.
- `--direct-device`: Device for the direct baseline evaluation (default `cpu`).
- `--direct-epochs`: Epochs for the logistic-regression probe trained on the
  source representations (default `200`).

### 3. Inspect the outputs

The script creates a timestamped directory under `experiments/forgetting/`
containing:

- `forgetting_matrix.csv`: Rows are source datasets, columns are `direct`
  (no transfer) plus each fine-tuning target. Values are accuracies evaluated on
  d1.
- `runs.jsonl`: Per-target metadata including GPU allocation and raw scores.
- `{dataset}_stdout.txt` / `{dataset}_stderr.txt`: Logs from each transfer run
  (including the `SourceEval` metric parsed for the forgetting matrix).
- `meta.json`: High-level configuration for reproducibility.

You can rerun the command with a different `--pretrained-model` or `--d2` list
to extend the CSV with new experiments as needed.
