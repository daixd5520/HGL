#!/usr/bin/env python3
"""
演示如何使用生成的embeddings进行可视化
"""

import os
import sys
sys.path.append('/mnt/data1/Graph/GRACE')

# 导入可视化脚本
from visualize_embeddings import visualize_embeddings

def main():
    # 设置路径
    base_dir = '/mnt/data1/Graph/HypGraphLoRA'
    embeddings_path = os.path.join(base_dir, 'embeddings', 'CiteSeer.GRACE.GAT.True_embeddings.npy')
    labels_path = os.path.join(base_dir, 'embeddings', 'CiteSeer.GRACE.GAT.True_labels.npy')
    save_path = os.path.join(base_dir, 'embeddings', 'CiteSeer.GRACE.GAT.True_visualization.png')

    print("Visualizing CiteSeer embeddings...")
    print(f"Embeddings: {embeddings_path}")
    print(f"Labels: {labels_path}")
    print(f"Save to: {save_path}")

    # 调用可视化函数
    visualize_embeddings(
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        save_path=save_path
    )

if __name__ == '__main__':
    main()

