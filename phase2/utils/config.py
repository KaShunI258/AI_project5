# 文件路径: phase1/utils/config.py
import torch
import os

class Config:
    def __init__(self, args):
        # ================= 路径配置 =================
        self.name = args.name
        self.data_dir = args.data_dir
        self.train_file = args.train_file
        self.test_file = args.test_file
        self.result_file = args.result_file
        
        self.text_model_name = args.text_model_name
        self.image_model_name = args.image_model_name
        
        # ================= 训练参数 =================
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.val_ratio = args.val_ratio
        self.early_stop_patience = args.early_stop_patience
        self.dropout = args.dropout
        
        # ================= 模型参数 =================
        self.use_text = args.use_text
        self.use_image = args.use_image
        self.feature_fusion = args.feature_fusion
        self.text_dim = args.text_dim
        self.image_dim = args.image_dim
        self.num_classes = args.num_classes
        
        # ================= 环境与设备 (重点修改) =================
        self.loss_type = args.loss_type
        self.project_name = args.project_name
        self.log_iteration = args.log_iteration
        self.seed = 42
        
        # --- 强制 GPU 逻辑 ---
        if torch.cuda.is_available():
            # 默认使用第一块可见的 GPU
            self.device = torch.device("cuda")
            # 获取当前 GPU 名称，方便确认
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n✅ GPU Detected: {gpu_name}")
            print(f"✅ Training will run on: {self.device}\n")
        else:
            self.device = torch.device("cpu")
            print("\n⚠️  WARNING: No GPU detected! Training will run on CPU. This will be slow.\n")