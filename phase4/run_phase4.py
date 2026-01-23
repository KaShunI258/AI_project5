import torch
import pandas as pd
import os
import shutil
import random  # [新增] 用于随机采样
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoImageProcessor
from torchvision import transforms

from utils.config import Config
from utils.dataload import MultimodalDataset
from multimodel import MultimodalModel
from trainer import MultimodalTrainer
from augmentations import DataAugmenter, VLMAugmenter

# === 配置 ===
# 确保这里指向你刚才重新生成的 Phase 3 模型路径
PHASE3_BEST_MODEL = "../phase3/results/Exp2_ACB_Loss_best.pth" 
DATA_DIR = "../dataset/data"
TRAIN_FILE = "../dataset/train_cleaned.txt"
NEW_TRAIN_FILE = "../dataset/train_phase4_augmented.txt"

def identify_hard_samples(config, model_path):
    """
    步骤 1: 使用 Phase 3 模型扫描训练集，找出 '难样本'
    定义：预测错误 或 正确但置信度低 (<0.6) 的样本
    """
    print("🔍 Scanning training set for Hard Samples...")
    
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    
    # 加载训练集 (不 shuffle, 方便索引对应)
    dataset = MultimodalDataset(config.data_dir, config.train_file, transform, is_train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = MultimodalModel(config).to(config.device)
    # 加载权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    model.eval()
    
    hard_samples = []
    
    with torch.no_grad():
        for i, (texts, images, labels) in enumerate(loader):
            texts = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(config.device)
            images = images.to(config.device)
            
            outputs = model(texts, images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            for j in range(len(labels)):
                true_label = labels[j].item()
                pred_label = preds[j].item()
                conf = confidences[j].item()
                
                global_idx = i * 32 + j
                if global_idx < len(dataset.df):
                    guid = str(dataset.df.iloc[global_idx]['guid'])
                    
                    # 判定条件：分错 或 (是Neutral但置信度低)
                    is_wrong = (true_label != pred_label)
                    is_weak_neutral = (true_label == 1 and conf < 0.6)
                    
                    if is_wrong or is_weak_neutral:
                        hard_samples.append(guid)
                    
    print(f"Found {len(hard_samples)} total hard samples.")
    return hard_samples

def main():
    # === 完善后的参数配置 ===
    class Args:
        def __init__(self):
            # 路径
            self.data_dir = DATA_DIR
            self.train_file = TRAIN_FILE
            self.test_file = "../dataset/test_without_label.txt"
            self.result_file = "result_phase4.txt" # [Fix] 补全参数
            
            # 模型路径
            self.text_model_name = "../pretrained_models/bert-base-uncased"
            self.image_model_name = "../pretrained_models/swinv2-base-patch4-window8-256"
            
            # 训练参数
            self.batch_size = 32
            self.learning_rate = 5e-5
            self.num_epochs = 15
            self.dropout = 0.1
            self.early_stop_patience = 4 # [Fix] 补全参数
            self.val_ratio = 0.1         # [Fix] 补全参数
            
            # 模型结构
            self.feature_fusion = 'attention_combine'
            self.text_dim = 256
            self.image_dim = 256
            self.num_classes = 3
            self.use_text = 1
            self.use_image = 1
            
            # 策略参数
            self.loss_type = 'acb'
            self.use_sampler = False
            self.alpha = 1.0
            self.beta = 0.1
            self.neural_init_weight = 1.0
            
            # 其他
            self.wandb = False
            self.name = "Phase4_Augmentation"
            self.project_name = "Phase4"
            self.log_iteration = 10

    args = Args()
    config = Config(args)
    
    # 1. 识别 Bad Cases
    hard_guids = identify_hard_samples(config, PHASE3_BEST_MODEL)
    
    # === [关键修改] 随机采样 25% ===
    sample_ratio = 0.25
    num_to_select = int(len(hard_guids) * sample_ratio)
    # 至少选 1 个，防止报错
    num_to_select = max(1, num_to_select)
    
    print(f"📉 Downsampling: Selecting {num_to_select} samples ({sample_ratio*100}%) from {len(hard_guids)} hard cases due to API limits.")
    
    selected_guids = random.sample(hard_guids, num_to_select)
    # ============================
    
    # 2. 执行 VLM 增强
    # ⚠️ 请在这里填入你真实的 Key ⚠️
    API_KEY = "sk-ee3a6bcdb0e442be9259d84599b03675" 
    
    vlm_augmenter = VLMAugmenter(api_key=API_KEY)
    vlm_augmenter.augment_dataset(selected_guids, DATA_DIR, NEW_TRAIN_FILE)
    
    # 3. 使用增强后的数据集重新训练
    print("\n🚀 Retraining with Augmented Data...")
    config.train_file = NEW_TRAIN_FILE # 切换为新数据集
    
    # 重新初始化组件
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    
    # 重新构建模型
    model = MultimodalModel(config)
    
    # 选择微调策略：加载 Phase 3 权重作为起点
    print(f"Loading weights from {PHASE3_BEST_MODEL} for finetuning...")
    if os.path.exists(PHASE3_BEST_MODEL):
        model.load_state_dict(torch.load(PHASE3_BEST_MODEL))
    else:
        print("Warning: Phase 3 weights not found, training from scratch.")
    
    trainer = MultimodalTrainer(model, tokenizer, config)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    
    full_dataset = MultimodalDataset(config.data_dir, config.train_file, transform, is_train=True)
    
    # 划分验证集
    from sklearn.model_selection import StratifiedShuffleSplit
    import numpy as np
    labels = full_dataset.labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.val_ratio, random_state=config.seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    trainer.train(train_loader, val_loader)
    
    # # 结束前强制保存 Phase 4 最终模型
    # torch.save(model.state_dict(), "results/Phase4_Best.pth")
    # print("Phase 4 Done. Model saved to results/Phase4_Best.pth")

if __name__ == "__main__":
    main()