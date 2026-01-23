# 文件路径: phase1/visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 设置风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文显示 (如果不需要可删)
plt.rcParams['axes.unicode_minus'] = False

# 配置
RESULT_DIR = "results"
EXP_NAME = "Phase1_Baseline" # 需与 main.py 中的 --name 一致
SAVE_DIR = "figures"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def plot_training_curves():
    """Fig 1-1: 训练曲线 (Loss & F1)"""
    log_path = os.path.join(RESULT_DIR, f"{EXP_NAME}_history.csv")
    if not os.path.exists(log_path):
        print(f"未找到日志文件: {log_path}")
        return
    
    df = pd.read_csv(log_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1: Loss
    sns.lineplot(data=df, x='epoch', y='train_loss', label='Train Loss', ax=ax1, marker='o')
    sns.lineplot(data=df, x='epoch', y='val_loss', label='Val Loss', ax=ax1, marker='o')
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    
    # 子图2: F1-Score
    sns.lineplot(data=df, x='epoch', y='train_f1', label='Train Weighted-F1', ax=ax2, marker='o')
    sns.lineplot(data=df, x='epoch', y='val_f1', label='Val Weighted-F1', ax=ax2, marker='o')
    ax2.set_title("Training & Validation F1-Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Weighted F1")
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'Fig1-1_Training_Curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close()

def plot_confusion_matrix_and_metrics():
    """
    Fig 1-2: 混淆矩阵
    Fig 1-3: 每类指标柱状图
    """
    pred_path = os.path.join(RESULT_DIR, f"{EXP_NAME}_val_preds.csv")
    if not os.path.exists(pred_path):
        print(f"未找到预测文件: {pred_path}")
        return
        
    df = pd.read_csv(pred_path)
    y_true = df['true_label']
    y_pred = df['pred_label']
    labels = [0, 1, 2]
    label_names = ['Positive', 'Neutral', 'Negative']
    
    # --- Fig 1-2: Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # 归一化 (显示百分比)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix ({EXP_NAME})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path_cm = os.path.join(SAVE_DIR, 'Fig1-2_Confusion_Matrix.png')
    plt.savefig(save_path_cm, dpi=300)
    print(f"Saved {save_path_cm}")
    plt.close()
    
    # --- Fig 1-3: Per-class Metrics ---
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    
    # 提取 precision, recall, f1
    metrics_data = []
    for cls in label_names:
        metrics_data.append({
            'Class': cls, 'Metric': 'Precision', 'Value': report[cls]['precision']
        })
        metrics_data.append({
            'Class': cls, 'Metric': 'Recall', 'Value': report[cls]['recall']
        })
        metrics_data.append({
            'Class': cls, 'Metric': 'F1-Score', 'Value': report[cls]['f1-score']
        })
        
    df_metrics = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metrics, x='Class', y='Value', hue='Metric', palette='viridis')
    plt.title('Per-class Performance Metrics')
    plt.ylim(0, 1.05)
    # 添加数值标签
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.2f', padding=3)
        
    save_path_bar = os.path.join(SAVE_DIR, 'Fig1-3_Class_Metrics.png')
    plt.savefig(save_path_bar, dpi=300)
    print(f"Saved {save_path_bar}")
    plt.close()

if __name__ == "__main__":
    plot_training_curves()
    plot_confusion_matrix_and_metrics()