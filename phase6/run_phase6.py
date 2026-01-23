import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from transformers import BertTokenizer, AutoImageProcessor
from tqdm import tqdm
from PIL import Image  # <--- [新增] 必须导入 PIL Image，因为 evaluate_clip_model 中用到了

# === 路径 Hack: 添加 phase4 到 sys.path 以便导入其模块 ===
# 当前文件所在目录: .../AI_course_of_ECNU/phase6
current_dir = os.path.dirname(os.path.abspath(__file__)) 
# 项目根目录: .../AI_course_of_ECNU
project_root = os.path.dirname(current_dir) 
phase4_dir = os.path.join(project_root, "phase4")

if phase4_dir not in sys.path:
    sys.path.append(phase4_dir)

# 导入 Phase 4 组件
from utils.config import Config
from utils.dataload import MultimodalDataset
from multimodel import MultimodalModel

# 导入 Phase 6 组件
from clip_classifier import CLIPZeroShot
from visualization import plot_confusion_matrices, plot_metrics_comparison

def evaluate_phase4_model(config, model_path, val_loader):
    print(f"Evaluating Phase 4 Model (loading from {model_path})...")
    
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    model = MultimodalModel(config).to(config.device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=config.device))
    else:
        raise FileNotFoundError(f"Model path not found: {model_path}")
        
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, images, labels in tqdm(val_loader, desc="Phase 4 Inference"):
            # 文本 Tokenize
            inputs = tokenizer(list(texts), padding=True, truncation=True, 
                               max_length=128, return_tensors="pt").to(config.device)
            images = images.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(inputs, images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)

def evaluate_clip_model(model_path, val_dataset):
    print(f"Evaluating CLIP Zero-Shot Model (loading from {model_path})...")
    
    # 初始化 CLIP
    clip_model = CLIPZeroShot(model_path)
    
    candidate_labels = ["positive", "neutral", "negative"]
    
    all_preds = []
    all_labels = []
    
    batch_size = 32
    indices = range(len(val_dataset))
    
    print("Running CLIP inference...")
    for i in tqdm(range(0, len(val_dataset), batch_size), desc="CLIP Inference"):
        batch_indices = indices[i : i + batch_size]
        
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            # 获取原始 Dataset 中的索引
            if isinstance(val_dataset, Subset):
                global_idx = val_dataset.indices[idx]
                item = val_dataset.dataset.data_list[global_idx]
            else:
                item = val_dataset.data_list[idx]
                
            guid = item['guid']
            label = item['label']
            
            # 读取图片
            base_dataset = val_dataset.dataset if isinstance(val_dataset, Subset) else val_dataset
            data_dir = base_dataset.data_dir
            
            img_path = os.path.join(data_dir, f"{guid}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(data_dir, f"{guid}.png")
            
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                # 如果图片损坏，给一张黑图防止报错
                image = Image.new('RGB', (224, 224), (0, 0, 0))
                
            batch_images.append(image)
            batch_labels.append(label)
            
        # 预测
        preds, _ = clip_model.predict(batch_images, candidate_labels)
        all_preds.extend(preds)
        all_labels.extend(batch_labels)
        
    return np.array(all_preds), np.array(all_labels)

class MockArgs:
    """模拟 argparse 传入配置"""
    def __init__(self):
        self.name = "Phase6_Comparison"
        
        # === 数据路径设置 (使用绝对路径更稳健) ===
        self.data_dir = os.path.join(project_root, "dataset", "data")
        self.train_file = os.path.join(project_root, "dataset", "train_cleaned.txt")
        self.test_file = os.path.join(project_root, "dataset", "test_without_label.txt")
        self.result_file = "result_phase6.txt"
        
        # === [修改点 1] 预训练模型路径 (BERT & Swin) ===
        # 对应路径: AI_course_of_ECNU/pretrained_models/bert-base-uncased
        self.text_model_name = os.path.join(project_root, "pretrained_models", "bert-base-uncased")
        
        # 对应路径: AI_course_of_ECNU/pretrained_models/swinv2-base-patch4-window8-256
        self.image_model_name = os.path.join(project_root, "pretrained_models", "swinv2-base-patch4-window8-256")
        
        # 参数需与 Phase 4 一致
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.num_epochs = 1
        self.val_ratio = 0.1
        self.early_stop_patience = 4
        self.dropout = 0.1
        self.use_text = 1
        self.use_image = 1
        self.feature_fusion = 'attention_combine'
        self.text_dim = 256
        self.image_dim = 256
        self.num_classes = 3
        
        # 其他
        self.loss_type = 'acb'
        self.wandb = False
        self.project_name = "Phase6"
        self.log_iteration = 10

def main():
    # 0. 准备输出目录
    results_dir = os.path.join(current_dir, "results")
    figures_dir = os.path.join(current_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. 准备数据 (Seed 42 划分)
    args = MockArgs()
    config = Config(args)
    
    # 强制覆盖 Config 中的路径，确保使用 MockArgs 中的绝对路径
    config.data_dir = args.data_dir
    config.train_file = args.train_file
    config.text_model_name = args.text_model_name
    config.image_model_name = args.image_model_name
    
    print(f"Data Source: {config.train_file}")
    
    # 加载数据集 (Phase 4 格式)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    
    full_dataset = MultimodalDataset(config.data_dir, config.train_file, transform, is_train=True)
    labels = full_dataset.labels
    
    # 划分验证集
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.val_ratio, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    # Val Loader (Phase 4 使用)
    val_dataset = Subset(full_dataset, val_idx)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    print(f"Validation Set Size: {len(val_dataset)}")
    
    # 2. 评估 Phase 4 模型
    phase4_model_path = os.path.join(phase4_dir, "results", "Phase4_Augmentation_best.pth")
    p4_preds, p4_labels = evaluate_phase4_model(config, phase4_model_path, val_loader)
    
    p4_acc = accuracy_score(p4_labels, p4_preds)
    p4_f1 = f1_score(p4_labels, p4_preds, average='weighted')
    print(f"Phase 4 Results -> Acc: {p4_acc:.4f}, F1: {p4_f1:.4f}")
    
    # 3. 评估 CLIP 模型
    # === [修改点 2] CLIP 模型路径 ===
    # 对应路径: AI_course_of_ECNU/phase6/protrained_models/clip-vit-large-patch14-336
    # 注意: 你提供的路径里是 protrained_models (不是 pretrained)，这里严格对应你的输入
    clip_model_path = os.path.join(current_dir, "protrained_models", "clip-vit-large-patch14-336")
    
    if not os.path.exists(clip_model_path):
        # 如果 protrained 写错了，尝试一下标准的 pretrained
        print(f"Warning: path {clip_model_path} not found, trying 'pretrained_models'...")
        clip_model_path = os.path.join(current_dir, "pretrained_models", "clip-vit-large-patch14-336")

    clip_preds, clip_labels = evaluate_clip_model(clip_model_path, val_dataset)
    
    clip_acc = accuracy_score(clip_labels, clip_preds)
    clip_f1 = f1_score(clip_labels, clip_preds, average='weighted')
    print(f"CLIP Results -> Acc: {clip_acc:.4f}, F1: {clip_f1:.4f}")
    
    # 4. 保存结果和可视化
    # 确保 label 顺序一致
    assert np.array_equal(p4_labels, clip_labels), "Label mismatch between evaluations!"
    
    metrics_p4 = {'Accuracy': p4_acc, 'F1': p4_f1}
    metrics_clip = {'Accuracy': clip_acc, 'F1': clip_f1}
    
    # 导出 Metrics CSV
    df_metrics = pd.DataFrame([metrics_p4, metrics_clip], index=['Phase4', 'CLIP'])
    df_metrics.to_csv(os.path.join(results_dir, "comparison_metrics.csv"))
    
    # 可视化
    plot_metrics_comparison(metrics_p4, metrics_clip, figures_dir)
    
    labels_names = ['positive', 'neutral', 'negative']
    plot_confusion_matrices(p4_labels, p4_preds, clip_preds, labels_names, figures_dir)
    
    print("\n✅ Phase 6 Comparison Completed!")
    print(f"Metrics saved to {results_dir}")
    print(f"Figures saved to {figures_dir}")

if __name__ == "__main__":
    main()