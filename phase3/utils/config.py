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
        self.output_dir = "results" # Phase 3 结果目录
        
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
        
        # ================= Phase 3 新增关键参数 (Fix Bug) =================
        # 必须把 main.py 里的新参数接进来
        self.loss_type = getattr(args, 'loss_type', 'ce')
        self.use_sampler = getattr(args, 'use_sampler', False)
        
        # ACB Loss 的超参数 (如果有默认值则读取，没有则设为默认)
        self.alpha = getattr(args, 'alpha', 1.0)
        self.beta = getattr(args, 'beta', 0.1)
        self.neural_init_weight = getattr(args, 'neural_init_weight', 1.0)
        
        # ================= 环境与设备 =================
        self.wandb = args.wandb
        self.project_name = args.project_name
        self.log_iteration = args.log_iteration
        self.seed = 42
        
        # --- 强制 GPU 逻辑 ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            # 这里的打印在多进程 DataLoader 时可能会刷屏，可以注释掉
            # print(f"Config: Running on {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Config: Running on CPU")