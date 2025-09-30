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

## Few-shot curvature stabilisation

Few-shot transfer now defaults to freezing the shared curvature parameter to the
pre-trained value in order to stabilise the backbone geometry. You can control
the behaviour via two new flags:

* `--few_curv_mode {auto,freeze,clone,regularize,none}` – `freeze` removes the
  curvature from the optimiser, `clone` keeps the backbone frozen but lets LoRA
  learn its own copy, `regularize` keeps curvature trainable with a quadratic
  penalty, and `none` reproduces the old behaviour. The default `auto` picks
  the value specified in `config.yaml` (currently `freeze`).
* `--few_curv_reg_lambda <float>` – strength of the quadratic penalty when
  `--few_curv_mode=regularize`. When omitted the value from `config.yaml` is
  used.

The active strategy and curvature values are logged to
`result/GraphLoRA.txt` for auditing few-shot experiments.
