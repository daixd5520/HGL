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

```
