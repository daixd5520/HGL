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
- `--repeats`: Number of independent repetitions. Each repetition runs the
  direct baseline plus every transfer and the script reports the mean ± standard
  deviation across repetitions (default `5`).

### 3. Inspect the outputs

The script creates a timestamped directory under `experiments/forgetting/`
containing:

- `forgetting_matrix.csv`: Rows are source datasets, columns are `direct`
  (no transfer) plus each fine-tuning target. Every cell contains
  `mean±std` (to four decimals) summarising accuracies on d1 over the configured
  repetitions.
- `runs.jsonl`: Per-run metadata including GPU allocation, raw scores, and a
  `repeat` index.
- `{dataset}_r{repeat}_stdout.txt` / `{dataset}_r{repeat}_stderr.txt`: Logs from
  each transfer run in repetition `repeat` (including the `SourceEval` metric
  parsed for the forgetting matrix).
- `meta.json`: High-level configuration for reproducibility.

You can rerun the command with a different `--pretrained-model` or `--d2` list
to extend the CSV with new experiments as needed.

## Curvature Sweep Experiment

To study how the initial manifold curvature impacts downstream transfer, use
`experiments/curvature_sweep.py`. The script automatically:

1. Pre-trains a fresh encoder for each curvature in the sweep (defaults span
   `0.1 → 10.0`, covering very flat to highly curved manifolds on a log scale).
2. Saves each checkpoint with a descriptive filename that now includes
   `curv_<value>` plus the repetition tag for easy traceability.
3. Evaluates the frozen encoder directly on the source dataset with a logistic
   probe.
4. Transfers the model to every requested target dataset, repeating the full
   pipeline `--repeats` times and aggregating `mean±std` scores into a CSV.

### Example usage

```bash
python experiments/curvature_sweep.py \
  --dataset Cora \
  --targets CiteSeer PubMed Photo \
  --curvatures 0.1 0.3 1.0 3.0 10.0 \
  --repeats 5 \
  --pretrain-gpu 0 \
  --transfer-gpus 0 1
```

Key options:

- `--dataset`: Source dataset used for all pre-training runs.
- `--targets`: List of datasets to fine-tune on. Defaults to every other
  supported dataset if omitted.
- `--curvatures`: Initial curvature values to sweep. You can provide any
  positive floats; the defaults strike a balance between shallow (0.1),
  moderate (1.0), and highly curved (10.0) manifolds.
- `--repeats`: Number of independent repetitions per curvature setting (default
  `5`).
- `--pretrain-gpu`: GPU id for pre-training runs.
- `--transfer-gpus`: GPUs to cycle through for transfer fine-tuning runs.
- `--direct-device` / `--direct-epochs`: Control the logistic probe used for
  direct evaluation of the frozen encoder.

Each sweep creates a timestamped folder under `experiments/curvature/` with:

- `curvature_matrix.csv`: Rows labelled `curv_<value>` and columns for `direct`
  plus every target dataset, populated with `mean±std` accuracies.
- `runs.jsonl`: Detailed logs for every pre-train, direct, and transfer run.
- `{target}_r{repeat}_stdout.txt` / `{target}_r{repeat}_stderr.txt`: Raw logs
  produced by `main.py` for each transfer.
- `meta.json`: Captures the command, sweep parameters, and output paths.
