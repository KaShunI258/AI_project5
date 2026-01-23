import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoImageProcessor
from torchvision import transforms

# 引用本地模块 (确保 utils, multimodel 在当前或是父级目录可引用的位置)
import sys
sys.path.append("..") # 允许引用上一级目录的模块
from phase4.utils.config import Config
from phase4.utils.dataload import MultimodalDataset
from phase4.multimodel import MultimodalModel

# === 配置路径 ===
PHASE3_MODEL_PATH = "../phase3/results/Exp2_ACB_Loss_best.pth" # Phase 3 最佳模型
PHASE4_MODEL_PATH = "results/Phase4_Augmentation_best.pth"      # Phase 4 最佳模型 (根据你之前保存的文件名调整)
DATA_DIR = "../dataset/data"
TRAIN_FILE_ORIG = "../dataset/train_cleaned.txt"
TRAIN_FILE_AUG = "../dataset/train_phase4_augmented.txt"
VAL_FILE = "../dataset/test_without_label.txt" # 注意：这里为了画混淆矩阵，最好是用带标签的验证集。
# 如果没有单独的验证集文件，代码里会自动从 train_cleaned 切分验证集。

# 设置风格
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 防止中文乱码兼容性问题，用英文通用字体
plt.rcParams['axes.unicode_minus'] = False
SAVE_DIR = "figures"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

class Args:
    """模拟参数类，用于加载 Config"""
    def __init__(self, train_file=TRAIN_FILE_ORIG):
        self.data_dir = DATA_DIR
        self.train_file = train_file
        self.test_file = VAL_FILE
        self.result_file = "dummy.txt"
        self.text_model_name = "../pretrained_models/bert-base-uncased"
        self.image_model_name = "../pretrained_models/swinv2-base-patch4-window8-256"
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.num_epochs = 15
        self.dropout = 0.1
        self.early_stop_patience = 4
        self.val_ratio = 0.1
        self.feature_fusion = 'attention_combine'
        self.text_dim = 256
        self.image_dim = 256
        self.num_classes = 3
        self.use_text = 1
        self.use_image = 1
        self.loss_type = 'acb'
        self.use_sampler = False
        self.alpha = 1.0
        self.beta = 0.1
        self.neural_init_weight = 1.0
        self.wandb = False
        self.name = "Viz"
        self.project_name = "Phase4"
        self.log_iteration = 10

def get_model_and_loader(model_path, train_file, is_augmented=False):
    """加载模型和数据加载器"""
    args = Args(train_file)
    config = Config(args)
    
    # 1. 准备数据
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    
    # 加载完整数据集
    full_dataset = MultimodalDataset(config.data_dir, config.train_file, transform, is_train=True)
    
    # 这里的关键是：我们要画验证集的混淆矩阵，或者训练集的特征分布
    # 为了 Fig 4-3 (t-SNE)，我们需要训练集数据
    # 为了 Fig 4-2 (混淆矩阵)，我们需要验证集数据
    
    # 简单切分 (保持随机种子一致以对齐)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    loader = DataLoader(val_set, batch_size=32, shuffle=False) # 默认返回验证集 loader
    train_loader_for_tsne = DataLoader(train_set, batch_size=32, shuffle=True) # 用于 t-SNE 抽样
    
    # 2. 加载模型
    model = MultimodalModel(config).to(config.device)
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=config.device))
    else:
        print(f"Warning: Model not found at {model_path}, using random weights.")
    
    model.eval()
    return model, loader, train_loader_for_tsne, config

def plot_fig4_1():
    """Fig 4-1: 全阶段性能演进图 (柱状图)"""
    print("Generating Fig 4-1...")
    # 这里的数据基于你之前的实验日志手动录入
    data = [
        {'Phase': 'Phase 1\n(Baseline)', 'Metric': 'Weighted F1', 'Score': 0.678},
        {'Phase': 'Phase 1\n(Baseline)', 'Metric': 'Neutral F1', 'Score': 0.333},
        
        {'Phase': 'Phase 2\n(Architecture)', 'Metric': 'Weighted F1', 'Score': 0.687},
        {'Phase': 'Phase 2\n(Architecture)', 'Metric': 'Neutral F1', 'Score': 0.407},
        
        {'Phase': 'Phase 3\n(Loss Strategy)', 'Metric': 'Weighted F1', 'Score': 0.672},
        {'Phase': 'Phase 3\n(Loss Strategy)', 'Metric': 'Neutral F1', 'Score': 0.355},
        
        {'Phase': 'Phase 4\n(Data Augment)', 'Metric': 'Weighted F1', 'Score': 0.740},
        {'Phase': 'Phase 4\n(Data Augment)', 'Metric': 'Neutral F1', 'Score': 0.522},
    ]
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Phase', y='Score', hue='Metric', data=df, palette="viridis")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
        
    plt.title("Fig 4-1: Performance Evolution Across All Phases", fontsize=16, pad=20)
    plt.ylim(0, 0.85)
    plt.ylabel("F1 Score")
    plt.xlabel("")
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/Fig4-1_Performance_Evolution.png", dpi=300)
    plt.close()

def get_predictions(model, loader, config):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, images, labels in loader:
            tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
            texts = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(config.device)
            images = images.to(config.device)
            outputs = model(texts, images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return all_labels, all_preds

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pos', 'Neu', 'Neg'], yticklabels=['Pos', 'Neu', 'Neg'])
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_fig4_2():
    """Fig 4-2: 增强前后混淆矩阵对比"""
    print("Generating Fig 4-2...")
    
    # 加载 Phase 3 模型 (增强前)
    model_p3, val_loader_p3, _, cfg_p3 = get_model_and_loader(PHASE3_MODEL_PATH, TRAIN_FILE_ORIG)
    labels_p3, preds_p3 = get_predictions(model_p3, val_loader_p3, cfg_p3)
    cm_p3 = confusion_matrix(labels_p3, preds_p3)
    
    # 加载 Phase 4 模型 (增强后)
    model_p4, val_loader_p4, _, cfg_p4 = get_model_and_loader(PHASE4_MODEL_PATH, TRAIN_FILE_AUG) # 注意用新的 Config 加载新数据
    labels_p4, preds_p4 = get_predictions(model_p4, val_loader_p4, cfg_p4) # 验证集其实是一样的切分
    cm_p4 = confusion_matrix(labels_p4, preds_p4)
    
    # 绘图
    plot_confusion_matrix(cm_p3, "Fig 4-2a: Confusion Matrix (Phase 3 - Before Aug)", f"{SAVE_DIR}/Fig4-2a_CM_Phase3.png")
    plot_confusion_matrix(cm_p4, "Fig 4-2b: Confusion Matrix (Phase 4 - After VLM Aug)", f"{SAVE_DIR}/Fig4-2b_CM_Phase4.png")

def extract_features(model, loader, config, num_batches=10):
    """提取特征用于 t-SNE"""
    features = []
    labels = []
    
    # Hook 获取倒数第二层特征 (Fusion Layer Output)
    # 假设 multimodel.py 中 classifier 是 nn.Sequential
    # 我们 hook classifier[0] 的输入，即 combined_features
    # 或者直接修改 forward 返回 feature。这里用 hook 比较通用。
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0].detach() # Linear 的 input 就是 combined_features
        return hook

    # 注册 hook 到 classifier 的第一层 Linear
    handle = model.classifier[0].register_forward_hook(get_activation('fusion'))
    
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    
    count = 0
    with torch.no_grad():
        for texts, images, lbls in loader:
            if count >= num_batches: break # 只取一部分数据画图，太慢
            
            texts = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(config.device)
            images = images.to(config.device)
            _ = model(texts, images) # 前向传播触发 hook
            
            feat = activation['fusion'].cpu().numpy()
            features.extend(feat)
            labels.extend(lbls.numpy())
            count += 1
            
    handle.remove()
    return np.array(features), np.array(labels)

def plot_tsne_scatter(features, labels, title, save_path):
    print(f"Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    feat_2d = tsne.fit_transform(features)
    
    df = pd.DataFrame({'x': feat_2d[:,0], 'y': feat_2d[:,1], 'label': labels})
    label_map = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
    df['label'] = df['label'].map(label_map)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette={'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}, alpha=0.7, s=60)
    plt.title(title, fontsize=14)
    plt.legend(title='Class')
    plt.axis('off') # t-SNE 坐标轴无意义
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_fig4_3():
    """Fig 4-3: 特征空间 t-SNE 分布对比"""
    print("Generating Fig 4-3...")
    
    # 提取 Phase 3 特征
    model_p3, _, train_loader_p3, cfg_p3 = get_model_and_loader(PHASE3_MODEL_PATH, TRAIN_FILE_ORIG)
    feats_p3, lbls_p3 = extract_features(model_p3, train_loader_p3, cfg_p3, num_batches=15)
    
    # 提取 Phase 4 特征
    model_p4, _, train_loader_p4, cfg_p4 = get_model_and_loader(PHASE4_MODEL_PATH, TRAIN_FILE_AUG)
    feats_p4, lbls_p4 = extract_features(model_p4, train_loader_p4, cfg_p4, num_batches=15)
    
    plot_tsne_scatter(feats_p3, lbls_p3, "Fig 4-3a: Feature Space (Phase 3 - Baseline)", f"{SAVE_DIR}/Fig4-3a_TSNE_Phase3.png")
    plot_tsne_scatter(feats_p4, lbls_p4, "Fig 4-3b: Feature Space (Phase 4 - VLM Aug)", f"{SAVE_DIR}/Fig4-3b_TSNE_Phase4.png")

if __name__ == "__main__":
    # 按顺序生成所有图
    plot_fig4_1() # 性能对比柱状图
    
    try:
        plot_fig4_2() # 混淆矩阵
    except Exception as e:
        print(f"Skipping Fig 4-2 due to error (check model paths): {e}")
        
    try:
        plot_fig4_3() # t-SNE
    except Exception as e:
        print(f"Skipping Fig 4-3 due to error: {e}")
        
    print(f"\n✅ All figures saved to {SAVE_DIR}/")