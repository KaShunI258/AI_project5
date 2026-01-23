import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(y_true, y_pred_phase4, y_pred_clip, labels, save_dir):
    """
    并排绘制 Phase 4 和 CLIP 的混淆矩阵
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    cm_phase4 = confusion_matrix(y_true, y_pred_phase4)
    cm_clip = confusion_matrix(y_true, y_pred_clip)
    
    # 归一化 (可选，这里显示数量更直观，也可以改为 'true' 显示比例)
    # cm_phase4 = cm_phase4.astype('float') / cm_phase4.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_phase4, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=labels, yticklabels=labels)
    axes[0].set_title('Phase 4 (Fine-tuned VLM) Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    sns.heatmap(cm_clip, annot=True, fmt='d', cmap='Greens', ax=axes[1], 
                xticklabels=labels, yticklabels=labels)
    axes[1].set_title('Phase 6 (Zero-Shot CLIP) Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "comparison_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved confusion matrix comparison to {save_path}")
    plt.close()

def plot_metrics_comparison(metrics_phase4, metrics_clip, save_dir):
    """
    绘制 Accuracy 和 F1 的对比柱状图
    metrics: dict {'Accuracy': 0.xx, 'F1': 0.xx}
    """
    labels = ['Accuracy', 'Weighted F1']
    p4_values = [metrics_phase4['Accuracy'], metrics_phase4['F1']]
    clip_values = [metrics_clip['Accuracy'], metrics_clip['F1']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, p4_values, width, label='Phase 4 (Fine-tuned)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, clip_values, width, label='Phase 6 (CLIP Zero-Shot)', color='#55a868')
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # 美化背景
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    save_path = os.path.join(save_dir, "comparison_metrics_bar.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved metrics comparison to {save_path}")
    plt.close()
