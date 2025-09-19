conda activate graph && cd Graph/HypGraphLoRA && systemctl start nvidia-fabricmanager


conda activate graph && systemctl start nvidia-fabricmanager
export CUDA_VISIBLE_DEVICES=1,0,2

conda activate graph && systemctl start nvidia-fabricmanager && cd Graph/HypGraphLoRA/
export CUDA_VISIBLE_DEVICES=3,4,5


conda activate graph && systemctl start nvidia-fabricmanager && export CUDA_VISIBLE_DEVICES=1,0,2