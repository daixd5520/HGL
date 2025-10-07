#!/usr/bin/env python3
"""
比较三个不同模型的CiteSeer embeddings t-SNE可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def load_embeddings_and_labels(emb_path, labels_path=None):
    """加载embeddings和labels"""
    embeddings = np.load(emb_path)
    if labels_path and os.path.exists(labels_path):
        labels = np.load(labels_path)
    else:
        labels = np.random.randint(0, 6, size=embeddings.shape[0])  # 随机标签用于测试

    return embeddings, labels

def create_comparison_tsne():
    """创建三个embeddings的t-SNE比较可视化"""

    # 文件路径
    files = [
        {
            'emb': '/mnt/data1/Graph/HypGraphLoRA/embeddings/CiteSeer.GRACE.GAT.hyp_True.True.20250925-191716_embeddings.npy',
            'labels': '/mnt/data1/Graph/HypGraphLoRA/embeddings/CiteSeer.GRACE.GAT.True_labels.npy',  # 使用相同的labels
            'title': 'GRACE'
        },
        {
            'emb': '/mnt/data1/Graph/HypGraphLoRA/embeddings/CiteSeer.GRACE.GAT.True_embeddings.npy',
            'labels': '/mnt/data1/Graph/HypGraphLoRA/embeddings/CiteSeer.GRACE.GAT.True_labels.npy',
            'title': 'GraphLoRA'
        },
        {
            'emb': '/mnt/data1/Graph/GRACE/embeddings_CiteSeer.npy',
            'labels': '/mnt/data1/Graph/GRACE/labels_CiteSeer.npy',
            'title': 'HGL'
        }
    ]

    # 配色方案 - 使用干净的同色系颜色
    colors = [
        '#8B7E99',  # 紫色系 - Class 0
        '#6B9BC3',  # 蓝色系 - Class 1
        '#C88B86',  # 粉色系 - Class 2
        '#8BAA7A',  # 绿色系 - Class 3
        '#D4A574',  # 棕色系 - Class 4
        '#7E9BB6'   # 灰蓝色系 - Class 5
    ]

    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    # fig.suptitle('CiteSeer Embeddings Comparison (t-SNE)', fontsize=16, fontweight='bold')

    # 存储所有embeddings_2d用于计算全局坐标范围
    all_embeddings_2d = []

    # 为每个模型生成t-SNE
    for i, file_info in enumerate(files):
        print(f"Processing {file_info['title']}...")

        # 加载数据
        embeddings, labels = load_embeddings_and_labels(file_info['emb'], file_info['labels'])

        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Unique labels: {np.unique(labels)}")

        # t-SNE降维 (使用相同的随机种子以确保可比性)
        tsne = TSNE(n_components=2, random_state=73, perplexity=20, max_iter=500)
        embeddings_2d = tsne.fit_transform(embeddings)
        all_embeddings_2d.append(embeddings_2d)

    # 计算全局坐标范围
    all_x_coords = np.concatenate([emb[:, 0] for emb in all_embeddings_2d])
    all_y_coords = np.concatenate([emb[:, 1] for emb in all_embeddings_2d])

    all_x_min, all_x_max = all_x_coords.min(), all_x_coords.max()
    all_y_min, all_y_max = all_y_coords.min(), all_y_coords.max()

    # 绘制每个子图
    for i, (file_info, embeddings_2d) in enumerate(zip(files, all_embeddings_2d)):
        # 加载数据（再次加载是为了获取labels）
        embeddings, labels = load_embeddings_and_labels(file_info['emb'], file_info['labels'])

        # 绘制子图
        ax = axes[i]

        # 如果标签是one-hot编码，转换为类别标签
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)

        # 绘制散点图
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=[colors[int(label) % len(colors)] for label in labels],
                           alpha=0.5, s=13, edgecolors='none')

        # 设置标题和标签
        ax.set_title(file_info['title'], fontsize=14, fontweight='bold')
        # ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
        # ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # 移除网格
        ax.grid(False)

    # 设置所有子图使用相同的坐标范围
    margin = 0.05
    # x_range = all_x_max - all_x_min
    # y_range = all_y_max - all_y_min
    # 不显示x轴和y轴
    ax.set_xticks([])
    ax.set_yticks([])

    # for ax in axes:
    #     ax.set_xlim(all_x_min - margin * x_range, all_x_max + margin * x_range)
    #     ax.set_ylim(all_y_min - margin * y_range, all_y_max + margin * y_range)

    # 添加颜色条标签
    legend_elements = []
    for j, color in enumerate(colors[:6]):  # 只显示前6个颜色
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=1,
                                        label=f'Class {j}'))

    # fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5),
    #           title='Classes', title_fontsize=12)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)  # 为图例留出空间

    # 保存图像
    output_path = '/mnt/data1/Graph/HypGraphLoRA/embeddings_comparison_tsne_2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison visualization saved to: {output_path}")

    # 显示图像
    plt.show()

if __name__ == '__main__':
    create_comparison_tsne()
