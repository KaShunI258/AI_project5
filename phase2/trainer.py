# 文件路径: phase1/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd  # 新增
import os            # 新增
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch.nn.functional as F

class MultimodalTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        
        self.model.to(self.device)
        
        # 确保输出目录存在
        self.output_dir = "results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 优化器: 只训练未冻结的参数
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=config.learning_rate, 
            weight_decay=0.01
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2,
        )
        
        self.early_stop_counter = 0
        self.best_model_state = None

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        preds_list = []
        labels_list = []
        
        for batch_idx, (texts, images, labels) in enumerate(train_loader):
            encoded_texts = self.tokenizer(
                list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(encoded_texts, images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            

        avg_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(labels_list, preds_list)
        epoch_f1 = f1_score(labels_list, preds_list, average='weighted')
        
        return avg_loss, epoch_acc, epoch_f1

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        
        # 1. 必须在这里初始化列表
        preds_list = []
        labels_list = []
        
        with torch.no_grad():
            for texts, images, labels in val_loader:
                # 数据移至设备
                encoded_texts = self.tokenizer(
                    list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(encoded_texts, images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # 获取预测结果
                preds = torch.argmax(outputs, dim=1)
                
                # 2. 必须在这里将当前 batch 的结果添加到列表中
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        
        # 计算平均 Loss
        avg_loss = total_loss / len(val_loader)
        
        # 3. 现在有了完整列表，可以安全计算指标了
        val_acc = accuracy_score(labels_list, preds_list)
        val_weighted_f1 = f1_score(labels_list, preds_list, average='weighted')
        val_macro_f1 = f1_score(labels_list, preds_list, average='macro')
        
        # 获取 Neutral (Label=1) 的 F1
        # 注意：target_names 必须与标签顺序对应 [0, 1, 2]
        report = classification_report(labels_list, preds_list, output_dict=True, zero_division=0)
        # 假设 label 1 是 neutral (根据 label map: positive:0, neutral:1, negative:2)
        # report 的键通常是字符串 '0', '1', '2'
        val_neutral_f1 = report['1']['f1-score'] 
        
        # 返回详细字典
        metrics = {
            'loss': avg_loss,
            'acc': val_acc,
            'weighted_f1': val_weighted_f1,
            'macro_f1': val_macro_f1,
            'neutral_f1': val_neutral_f1
        }
        
        print(f"Eval - Acc: {val_acc:.4f}, W-F1: {val_weighted_f1:.4f}, Neu-F1: {val_neutral_f1:.4f}")
        return metrics

    def train(self, train_loader, val_loader):
            print(f"Start Training on {self.device}...")
            cur_best_f1 = 0.0
            history = [] # 用于记录训练日志
            best_metrics = {} # 用于返回给自动化脚本

            for epoch in range(self.config.num_epochs):
                print(f"\n=== Epoch {epoch+1}/{self.config.num_epochs} ===")
                
                # 1. 训练
                t_loss, t_acc, t_f1 = self.train_epoch(train_loader, epoch)
                
                # 2. 验证 (只调用一次！)
                metrics = self.evaluate(val_loader)
                
                # 从字典中提取指标
                v_loss = metrics['loss']
                v_acc = metrics['acc']
                v_f1 = metrics['weighted_f1']
                
                print(f"[Train] Loss: {t_loss:.4f}, Acc: {t_acc:.4f}, F1: {t_f1:.4f}")
                # evaluate 内部已经打印了验证集信息，这里不需要重复打印详细信息
                
                # === 记录日志 ===
                history.append({
                    'epoch': epoch + 1,
                    'train_loss': t_loss, 'train_acc': t_acc, 'train_f1': t_f1,
                    'val_loss': v_loss, 'val_acc': v_acc, 'val_f1': v_f1,
                    'val_neutral_f1': metrics['neutral_f1'] # 顺便记录一下 neutral F1
                })
                
                # 3. 学习率调整
                self.scheduler.step(v_f1)
                
                # 4. 早停与保存最佳模型
                if v_f1 > cur_best_f1:
                    cur_best_f1 = v_f1
                    best_metrics = metrics # 更新最佳指标字典
                    self.best_model_state = deepcopy(self.model.state_dict())
                    
                    # 保存权重 (可选，为了节省空间可以只存 best_model.pth，或者按 name 存)
                    save_path = f"{self.output_dir}/{self.config.name}_best.pth"
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Saving Best Model (F1: {cur_best_f1:.4f}) to {save_path}")
                    
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    
                if self.early_stop_counter >= self.config.early_stop_patience:
                    print("Early Stopping Triggered.")
                    break
            
            # === 循环结束后：保存训练日志 CSV ===
            # 这个文件对分析单个模型的训练过程非常重要，必须保留
            df_history = pd.DataFrame(history)
            log_path = os.path.join(self.output_dir, f"{self.config.name}_history.csv")
            df_history.to_csv(log_path, index=False)
            print(f"Training log saved to {log_path}")
            
            # === 训练结束：使用最佳模型生成验证集详细预测（用于混淆矩阵）===
            # 这个文件对画混淆矩阵非常重要，必须保留
            if self.best_model_state is not None:
                print("Generating validation predictions for confusion matrix...")
                self.model.load_state_dict(self.best_model_state)
                self.save_val_predictions(val_loader)

            # 返回最佳指标，供 run_phase2.py 汇总使用
            return best_metrics

    def save_val_predictions(self, val_loader):
        """保存验证集的详细预测结果 (True vs Pred)"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for texts, images, labels in val_loader:
                encoded_texts = self.tokenizer(
                    list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)
                images = images.to(self.device)
                
                outputs = self.model(encoded_texts, images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 保存到 CSV
        df_val = pd.DataFrame({
            'true_label': all_labels,
            'pred_label': all_preds
        })
        val_path = os.path.join(self.output_dir, f"{self.config.name}_val_preds.csv")
        df_val.to_csv(val_path, index=False)
        print(f"Validation predictions saved to {val_path}")

    def predict(self, test_loader):
        # 加载最佳权重
        self.model.load_state_dict(torch.load(f"{self.output_dir}/{self.config.name}_best.pth"))
        self.model.eval()
        results = {}
        label_map_rev = {0: 'positive', 1: 'neutral', 2: 'negative'}
        
        with torch.no_grad():
            for texts, images, guids in test_loader:
                encoded_texts = self.tokenizer(
                    list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self.device)
                images = images.to(self.device)
                
                outputs = self.model(encoded_texts, images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                for guid, pred in zip(guids, preds):
                    results[guid] = label_map_rev[pred]
        return results