# 文件路径: phase1/main.py
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, AutoImageProcessor
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random
import os

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
    # 1. 参数定义 (Phase 1 Baseline Default)
    parser = argparse.ArgumentParser()
    # 路径参数 (向上两级)
    parser.add_argument('--data_dir', default='../dataset/data')
    parser.add_argument('--train_file', default='../dataset/train_cleaned.txt')
    parser.add_argument('--test_file', default='../dataset/test_without_label.txt')
    parser.add_argument('--result_file', default='result_phase1.txt')
    parser.add_argument('--text_model_name', default='../pretrained_models/bert-base-uncased')
    parser.add_argument('--image_model_name', default='../pretrained_models/swinv2-base-patch4-window8-256')
    
    # 训练超参
    parser.add_argument('--name', default='Phase1_Baseline')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4) # Peak LR
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    
    # 结构参数
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--use_image', type=int, default=1)
    parser.add_argument('--feature_fusion', default='concat')
    parser.add_argument('--text_dim', type=int, default=256)
    parser.add_argument('--image_dim', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=3)
    
    # 其他
    parser.add_argument('--loss_type', default='ce')
    parser.add_argument('--project_name', default='Multimodal_Phase1')
    parser.add_argument('--log_iteration', type=int, default=10)

    args = parser.parse_args()
    config = Config(args)
    set_seed(config.seed)
    
    # 2. 准备预处理
    print(f"Loading models from: {config.text_model_name} & {config.image_model_name}")
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    
    # 使用 Swin 推荐的归一化参数
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    
    # 3. 加载完整数据集
    full_dataset = MultimodalDataset(
        data_dir=config.data_dir,
        index_file=config.train_file,
        transform=transform,
        is_train=True
    )
    print(f"Total samples: {len(full_dataset)}")
    
    # 4. Stratified Split (分层划分)
    # 提取所有标签用于划分
    labels = full_dataset.labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.val_ratio, random_state=config.seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # 5. 初始化模型和训练
    model = MultimodalModel(config)
    trainer = MultimodalTrainer(model, tokenizer, config)
    
    # 开始训练
    trainer.train(train_loader, val_loader)
    
    # 6. 预测测试集
    print("\nPredicting test set...")
    test_dataset = MultimodalDataset(
        data_dir=config.data_dir,
        index_file=config.test_file,
        transform=transform,
        is_train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    results = trainer.predict(test_loader)
    
    # 写入结果
    with open(config.result_file, 'w') as f:
        f.write("guid,tag\n")
        for guid, tag in results.items():
            f.write(f"{guid},{tag}\n")
    print(f"Results saved to {config.result_file}")

if __name__ == "__main__":
    main()