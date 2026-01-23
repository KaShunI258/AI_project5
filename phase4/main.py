import argparse
import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from transformers import BertTokenizer, AutoImageProcessor
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

# 引用同级模块
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
    # Phase 3 关键参数
    parser.add_argument('--loss_type', default='ce', choices=['ce', 'acb'], help='CrossEntropy or AdaptiveClassBalanced')
    parser.add_argument('--use_sampler', default='False', help='Use WeightedRandomSampler')
    
    # 固定最优架构
    parser.add_argument('--feature_fusion', default='attention_combine')
    parser.add_argument('--text_dim', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=5e-5) # 可以用你搜索出的最优LR
    
    # 基础参数
    parser.add_argument('--name', default='Phase3_Exp')
    parser.add_argument('--data_dir', default='../dataset/data')
    parser.add_argument('--train_file', default='../dataset/train_cleaned.txt')
    parser.add_argument('--test_file', default='../dataset/test_without_label.txt')
    parser.add_argument('--text_model_name', default='../pretrained_models/bert-base-uncased')
    parser.add_argument('--image_model_name', default='../pretrained_models/swinv2-base-patch4-window8-256')
    parser.add_argument('--result_file', default='result.txt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--early_stop_patience', type=int, default=4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--use_image', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--wandb', default='False')
    parser.add_argument('--project_name', default='Phase3')
    parser.add_argument('--log_iteration', type=int, default=10)
    
    args = parser.parse_args()
    args.use_sampler = True if args.use_sampler.lower() == 'true' else False
    
    config = Config(args)
    config.output_dir = "results"
    set_seed(config.seed)
    
    # 1. Data
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    full_dataset = MultimodalDataset(config.data_dir, config.train_file, transform, is_train=True)
    
    labels = full_dataset.labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.val_ratio, random_state=config.seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    # Sampler Logic
    sampler = None
    shuffle = True
    if config.use_sampler:
        print("Using WeightedRandomSampler for Imbalance...")
        train_labels = [full_dataset.labels[i] for i in train_idx]
        class_counts = np.bincount(train_labels)
        class_weights = 1. / class_counts
        sample_weights = [class_weights[l] for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False # Sampler 和 Shuffle 互斥
        
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=shuffle, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # 2. Model & Train
    model = MultimodalModel(config)
    trainer = MultimodalTrainer(model, tokenizer, config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()