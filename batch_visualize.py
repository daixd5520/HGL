#!/usr/bin/env python3
"""
批量生成所有CiteSeer模型embeddings的可视化
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import glob

def visualize_embeddings(embeddings_path, labels_path, save_path, title=None):
    """
    使用t-SNE可视化embeddings
    """
    # 加载embeddings和labels
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)

    print(f"Visualizing: {os.path.basename(embeddings_path)}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")

    # 使用t-SNE降维到2D
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 可视化
    plt.figure(figsize=(8, 6))

    # 如果标签是one-hot编码，转换为类别标签
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)

    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=labels, cmap='tab10', alpha=0.6, s=50)

    plt.colorbar(scatter)
    if title:
        plt.title(title)
    else:
        model_name = os.path.splitext(os.path.basename(embeddings_path))[0].replace('_embeddings', '')
        plt.title(f't-SNE: {model_name}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to: {save_path}")


def batch_visualize(embeddings_dir='/mnt/data1/Graph/HypGraphLoRA/embeddings',
                   labels_path=None):
    """
    批量生成所有embeddings的可视化
    """
    print("Starting batch visualization...")

    # 查找所有embeddings文件
    embeddings_pattern = os.path.join(embeddings_dir, '*_embeddings.npy')
    embeddings_files = glob.glob(embeddings_pattern)

    print(f"Found {len(embeddings_files)} embeddings files")

    # 如果没有指定labels路径，查找对应的labels文件
    if labels_path is None:
        labels_pattern = os.path.join(embeddings_dir, '*_labels.npy')
        labels_files = glob.glob(labels_pattern)
        if labels_files:
            labels_path = labels_files[0]  # 使用第一个找到的labels文件
            print(f"Using labels file: {labels_path}")
        else:
            print("No labels file found, will use random colors")
            labels_path = None

    # 为每个embeddings文件生成可视化
    for emb_file in embeddings_files:
        model_name = os.path.splitext(os.path.basename(emb_file))[0].replace('_embeddings', '')

        # 构造可视化保存路径
        viz_file = os.path.join(embeddings_dir, f"{model_name}_visualization.png")

        try:
            visualize_embeddings(emb_file, labels_path, viz_file)
        except Exception as e:
            print(f"Error visualizing {model_name}: {e}")

    print("Batch visualization completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch visualize embeddings')
    parser.add_argument('--embeddings_dir', type=str,
                       default='/mnt/data1/Graph/HypGraphLoRA/embeddings',
                       help='Directory containing embeddings files')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to labels file (optional)')

    args = parser.parse_args()

    batch_visualize(args.embeddings_dir, args.labels)

