# GraphLoRA
This is an official implementation of KDD 25 paper GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer Learning.

## 📰 News

 1. **2024-12**
🎉 Our paper *"GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer Learning"* has been accepted to the **KDD 2025**!
📄 [Read the paper on arXiv](https://arxiv.org/abs/2409.16670)

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
python main.py --is_pretrain True

# Fine-tuning
python main.py --is_transfer True
```
