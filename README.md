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
python main.py --is_pretrain True

# Fine-tuning
python main.py --is_transfer True
```
