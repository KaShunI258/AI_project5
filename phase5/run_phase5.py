import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, AutoImageProcessor
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

# 引用本地模块
from utils.config import Config
from utils.dataload import MultimodalDataset
from multimodel import MultimodalModel

# === 配置 ===
# 注意：这里严格使用你指定的文件名
PHASE4_BEST_MODEL = "../phase4/results/Phase4_Augmentation_best.pth"
DATA_DIR = "../dataset/data"
# 使用增强后的数据作为来源，但我们只关心验证集
TRAIN_FILE_AUG = "../dataset/train_phase4_augmented.txt" 

class Args:
    def __init__(self):
        # 基础配置保持与 Phase 4 一致，确保模型能加载
        self.data_dir = DATA_DIR
        self.train_file = TRAIN_FILE_AUG
        self.test_file = "dummy"
        self.result_file = "dummy"
        self.text_model_name = "../pretrained_models/bert-base-uncased"
        self.image_model_name = "../pretrained_models/swinv2-base-patch4-window8-256"
        self.batch_size = 32
        self.feature_fusion = 'attention_combine'
        self.text_dim = 256
        self.image_dim = 256
        self.num_classes = 3
        self.use_text = 1
        self.use_image = 1
        self.dropout = 0.1
        # 必须有的参数，防止 Config 报错
        self.learning_rate = 5e-5 
        self.num_epochs = 1
        self.val_ratio = 0.1
        self.early_stop_patience = 4
        self.loss_type = 'acb'
        self.use_sampler = False
        self.wandb = False
        self.name = "Phase5_Ablation"
        self.project_name = "Phase5"
        self.log_iteration = 10

def evaluate_model(model, loader, config, mode_name):
    """在指定消融模式下评估模型"""
    print(f"\n🔍 Evaluating Mode: [{mode_name}] ...")
    
    # 动态设置消融模式
    config.ablation_mode = mode_name
    model.config.ablation_mode = mode_name # 确保模型内部能读到
    
    model.eval()
    all_preds = []
    all_labels = []
    
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    
    with torch.no_grad():
        for texts, images, labels in loader:
            encoded_texts = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(config.device)
            images = images.to(config.device)
            
            # 模型 forward 会根据 config.ablation_mode 自动把对应特征置零
            outputs = model(encoded_texts, images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    # 计算详细指标
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 获取 Neutral (Label=1) 的 F1
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    neutral_f1 = report['1']['f1-score']
    
    print(f"   -> Acc: {acc:.4f} | F1: {f1:.4f} | Neu_F1: {neutral_f1:.4f}")
    
    return {
        "Modality": mode_name,
        "Accuracy": acc,
        "Precision": p,
        "Recall": r,
        "Weighted F1": f1,
        "Neutral F1": neutral_f1
    }

def main():
    print("🚀 Starting Phase 5: Modality Ablation Study...")
    os.makedirs("results", exist_ok=True)
    
    # 1. 准备数据和模型
    args = Args()
    config = Config(args)
    
    # 加载数据 (使用与 Phase 4 一致的划分方式，保证验证集是公平的)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    
    full_dataset = MultimodalDataset(config.data_dir, config.train_file, transform, is_train=True)
    labels = full_dataset.labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.val_ratio, random_state=42) # 固定种子42
    _, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    val_subset = Subset(full_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # 加载模型
    model = MultimodalModel(config).to(config.device)
    if os.path.exists(PHASE4_BEST_MODEL):
        print(f"✅ Loading weights from {PHASE4_BEST_MODEL}")
        model.load_state_dict(torch.load(PHASE4_BEST_MODEL, map_location=config.device))
    else:
        print(f"❌ Error: Model file not found at {PHASE4_BEST_MODEL}")
        return

    # 2. 执行三次实验
    results = []
    
    # (1) Text+Image (完整模型)
    res_full = evaluate_model(model, val_loader, config, mode_name="none") # none表示不消融
    res_full["Modality"] = "Text + Image (Full)" # 改个好听的名字
    results.append(res_full)
    
    # (2) Text-only (屏蔽图像)
    res_text = evaluate_model(model, val_loader, config, mode_name="text_only")
    res_text["Modality"] = "Text Only"
    results.append(res_text)
    
    # (3) Image-only (屏蔽文本)
    res_image = evaluate_model(model, val_loader, config, mode_name="image_only")
    res_image["Modality"] = "Image Only"
    results.append(res_image)
    
    # 3. 保存并展示表格
    df = pd.DataFrame(results)
    # 调整列顺序
    cols = ["Modality", "Accuracy", "Precision", "Recall", "Weighted F1", "Neutral F1"]
    df = df[cols]
    
    print("\n🏆 Tab5-1: Modality Ablation Table")
    print(df.to_string(index=False))
    
    df.to_csv("results/Tab5-1_Modality_Ablation.csv", index=False)
    print("\n✅ Results saved to results/Tab5-1_Modality_Ablation.csv")

if __name__ == "__main__":
    main()