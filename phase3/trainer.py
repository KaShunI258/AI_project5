import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import torch.nn.functional as F
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score, classification_report

class MultimodalTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.model.to(self.device)
        self.output_dir = config.output_dir if hasattr(config, 'output_dir') else "results"
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        # 优化器
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                     lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)
        
        # === ACB Loss 动态变量 ===
        self.class_difficulty = torch.zeros(config.num_classes).to(self.device) # 累积难度
        self.class_counts = torch.zeros(config.num_classes).to(self.device)     # 累积样本数
        self.criterion = nn.CrossEntropyLoss() # 备用基础Loss
        
        self.early_stop_counter = 0
        self.best_model_state = None

    def adaptive_class_balanced_loss(self, outputs, labels):
        """Phase 3 核心创新：自适应类别平衡损失"""
        # 1. 基础 CE
        ce_loss = F.cross_entropy(outputs, labels, reduction='none')
        
        # 2. 计算预测概率 (Pt)
        probs = F.softmax(outputs, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # 3. 动态更新难度 (无梯度)
        with torch.no_grad():
            batch_counts = torch.bincount(labels, minlength=self.config.num_classes)
            # 难度定义为 1-pt (预测越准难度越低)
            batch_diff = torch.bincount(labels, weights=(1-pt), minlength=self.config.num_classes)
            
            self.class_counts += batch_counts
            self.class_difficulty += batch_diff
            
            # 平均难度
            avg_diff = self.class_difficulty / (self.class_counts + 1e-8)
            # 动态权重: 难度越大，权重越高
            alpha_weights = 1.0 + torch.log(avg_diff + 1.0)
            # 动态 Focal Gamma
            gamma_dynamic = 1.0 + avg_diff[labels] 

        # 4. 计算 Focal Loss 部分
        focal_loss = (1 - pt).pow(gamma_dynamic) * ce_loss
        
        # 5. 边界增强项 (Boundary Term): 惩罚 Top1 和 Top2 概率过于接近
        # 目的：让 Neutral 和 Pos/Neg 的边界更清晰
        top2_probs, _ = torch.topk(probs, 2, dim=1)
        margin = top2_probs[:, 0] - top2_probs[:, 1]
        boundary_loss = torch.exp(-margin) # margin越小(边界模糊)，loss越大
        
        # 6. 总 Loss
        # alpha_weights 广播到 batch
        batch_weights = alpha_weights[labels]
        total_loss = (focal_loss * batch_weights).mean() + 0.1 * boundary_loss.mean()
        
        return total_loss

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        preds_list, labels_list = [], []
        
        for texts, images, labels in train_loader:
            encoded_texts = self.tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(encoded_texts, images)
            
            # === Switch Loss ===
            if self.config.loss_type == 'acb':
                loss = self.adaptive_class_balanced_loss(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(labels_list, preds_list)
        f1 = f1_score(labels_list, preds_list, average='weighted')
        return avg_loss, acc, f1

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        preds_list, labels_list = [], []
        
        with torch.no_grad():
            for texts, images, labels in val_loader:
                encoded_texts = self.tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(encoded_texts, images)
                loss = self.criterion(outputs, labels) # 验证集统一用 CE 方便对比 Loss 数值
                
                total_loss += loss.item()
                preds_list.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'acc': accuracy_score(labels_list, preds_list),
            'weighted_f1': f1_score(labels_list, preds_list, average='weighted'),
            'neutral_f1': classification_report(labels_list, preds_list, output_dict=True, zero_division=0)['1']['f1-score']
        }
        return metrics

    def train(self, train_loader, val_loader):
        print(f"Start Training (Loss: {self.config.loss_type})...")
        best_f1 = 0.0
        history, best_metrics = [], {}
        
        for epoch in range(self.config.num_epochs):
            t_loss, t_acc, t_f1 = self.train_epoch(train_loader, epoch)
            metrics = self.evaluate(val_loader)
            
            print(f"Ep {epoch+1} | T_Loss:{t_loss:.4f} T_F1:{t_f1:.4f} | V_Loss:{metrics['loss']:.4f} V_F1:{metrics['weighted_f1']:.4f} Neu_F1:{metrics['neutral_f1']:.4f}")
            
            history.append({
                'epoch': epoch+1, 'train_loss': t_loss, 'train_f1': t_f1,
                'val_loss': metrics['loss'], 'val_f1': metrics['weighted_f1'], 'neutral_f1': metrics['neutral_f1']
            })
            
            self.scheduler.step(metrics['weighted_f1'])
            
            if metrics['weighted_f1'] > best_f1:
                best_f1 = metrics['weighted_f1']
                best_metrics = metrics
                self.best_model_state = deepcopy(self.model.state_dict())
                self.early_stop_counter = 0

                # === [新增] 保存最佳模型权重到硬盘 ===
                save_path = os.path.join(self.output_dir, f"{self.config.name}_best.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"   >>> Model saved to {save_path}")
                # ===================================
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.config.early_stop_patience: break
        
        # Save History
        pd.DataFrame(history).to_csv(os.path.join(self.output_dir, f"{self.config.name}_history.csv"), index=False)
        
        # Save Predictions with Confidence (For Fig 3-3)
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.save_val_predictions(val_loader)
            
        return best_metrics

    def save_val_predictions(self, val_loader):
        self.model.eval()
        data = []
        with torch.no_grad():
            for texts, images, labels in val_loader:
                encoded_texts = self.tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                images = images.to(self.device)
                
                outputs = self.model(encoded_texts, images)
                probs = F.softmax(outputs, dim=1) # 转化为概率
                confidences, preds = torch.max(probs, dim=1) # 获取最大概率(置信度)和预测类别
                
                for t, p, c in zip(labels.cpu().numpy(), preds.cpu().numpy(), confidences.cpu().numpy()):
                    data.append({'true_label': t, 'pred_label': p, 'confidence': c})
                    
        pd.DataFrame(data).to_csv(os.path.join(self.output_dir, f"{self.config.name}_val_preds.csv"), index=False)