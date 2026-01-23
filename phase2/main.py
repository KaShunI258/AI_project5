import argparse
import torch
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, AutoImageProcessor
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random
import os

# 确保引用的是 phase2 目录下的模块
from utils.config import Config
from utils.dataload import MultimodalDataset
from multimodel import MultimodalModel
from trainer import MultimodalTrainer

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    
    # --- 核心搜索参数 ---
    parser.add_argument('--feature_fusion', type=str, default='attention_combine')
    parser.add_argument('--text_dim', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=256) # 实际上 multimodel 里会强制 image_dim=text_dim
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # --- 基础配置参数 ---
    parser.add_argument('--name', default='Debug_Exp')
    parser.add_argument('--data_dir', default='../dataset/data')
    parser.add_argument('--train_file', default='../dataset/train_cleaned.txt')
    parser.add_argument('--test_file', default='../dataset/test_without_label.txt')
    parser.add_argument('--text_model_name', default='../pretrained_models/bert-base-uncased')
    parser.add_argument('--image_model_name', default='../pretrained_models/swinv2-base-patch4-window8-256')
    parser.add_argument('--result_file', default='result_debug.txt')
    
    # --- 其他参数 (保持默认) ---
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--use_image', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--loss_type', default='ce')
    parser.add_argument('--wandb', type=str, default='False') # 接收字符串，避免 bool 解析错误
    parser.add_argument('--project_name', default='Phase2_Search')
    parser.add_argument('--log_iteration', type=int, default=10)

    args = parser.parse_args()
    
    # 处理 wandb 布尔值
    args.wandb = True if args.wandb.lower() == 'true' else False
    
    config = Config(args)
    set_seed(config.seed)
    
    # 1. 准备数据
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    
    full_dataset = MultimodalDataset(config.data_dir, config.train_file, transform, is_train=True)
    
    # 划分验证集
    labels = full_dataset.labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.val_ratio, random_state=config.seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # 2. 初始化模型
    model = MultimodalModel(config)
    trainer = MultimodalTrainer(model, tokenizer, config)
    
    # 3. 训练并返回最佳结果
    # 注意：这里 trainer.train 返回的是 best_metrics 字典
    best_metrics = trainer.train(train_loader, val_loader)
    
    # 4. 关键：打印特定的格式供 search_hyperparams.py 解析
    # search_hyperparams.py 会抓取 "Best validation accuracy: 0.xxxx"
    print(f"Best validation accuracy: {best_metrics['acc']:.4f}")
    print(f"Best validation F1: {best_metrics['weighted_f1']:.4f}")

if __name__ == "__main__":
    main()