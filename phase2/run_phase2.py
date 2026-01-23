import os
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, AutoImageProcessor
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random

# 引用同级目录下的模块
from utils.config import Config
from utils.dataload import MultimodalDataset
from multimodel import MultimodalModel
from trainer import MultimodalTrainer

# 融合阶梯列表
FUSION_METHODS = [
    'concat',            # 0-Order
    'combine',           # 0-Order
    'attention',         # 1-Order
    'attention_concat',  # 2-Order (H1)
    'attention_combine', # 2-Order
    'encoder'            # 3-Order
]

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_experiment(fusion_method, device_id=0):
    print(f"\n{'='*20} Running Phase 2: {fusion_method} {'='*20}")
    
    # 模拟命令行参数
    class Args:
        def __init__(self):
            self.name = f"Phase2_{fusion_method}"
            self.data_dir = '../dataset/data'
            self.train_file = '../dataset/train_cleaned.txt'
            self.test_file = '../dataset/test_without_label.txt'
            self.result_file = f'results/{self.name}_pred.txt'
            self.text_model_name = '../pretrained_models/bert-base-uncased'
            self.image_model_name = '../pretrained_models/swinv2-base-patch4-window8-256'
            
            # 保持和 Phase 1 一致的训练参数 (控制变量)
            self.batch_size = 32
            self.learning_rate = 1e-4
            self.num_epochs = 20     # 可以稍微减少，因为我们只看收敛趋势
            self.dropout = 0.2
            self.early_stop_patience = 5
            self.val_ratio = 0.1
            
            # 融合方式 (变量)
            self.use_text = 1
            self.use_image = 1
            self.feature_fusion = fusion_method
            self.text_dim = 512
            self.image_dim = 512
            self.num_classes = 3
            
            self.loss_type = 'ce'
            self.wandb = False       # 批量跑建议关闭 wandb，或者只记录不监控
            self.project_name = 'Phase2_Fusion_Ladder'
            self.log_iteration = 50

    args = Args()
    config = Config(args)
    set_seed(config.seed)
    
    # --- 数据加载 (同 Phase 1) ---
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    
    full_dataset = MultimodalDataset(config.data_dir, config.train_file, transform, is_train=True)
    labels = full_dataset.labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.val_ratio, random_state=config.seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # --- 模型与训练 ---
    model = MultimodalModel(config)
    trainer = MultimodalTrainer(model, tokenizer, config)
    
    # 这里的 train 需要稍微改一下 trainer.py 让它返回 best_val_metrics
    # 或者我们直接读取 log csv (为了简化，这里假设 trainer.train 返回最佳 F1)
    # *建议修改 trainer.py 的 train 方法返回 best_metrics 字典*
    best_metrics = trainer.train(train_loader, val_loader) 
    
    return best_metrics

def main():
    results = []
    
    if not os.path.exists("results"):
        os.makedirs("results")

    for method in FUSION_METHODS:
        try:
            metrics = run_experiment(method)
            metrics['fusion_method'] = method
            results.append(metrics)
        except Exception as e:
            print(f"Error running {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存汇总结果
    df = pd.DataFrame(results)
    # 调整列顺序
    cols = ['fusion_method', 'acc', 'weighted_f1', 'macro_f1', 'neutral_f1']
    df = df[cols]
    
    df.to_csv("phase2_summary.csv", index=False)
    print("\nPhase 2 Complete! Results saved to phase2_summary.csv")
    print(df)

if __name__ == "__main__":
    main()