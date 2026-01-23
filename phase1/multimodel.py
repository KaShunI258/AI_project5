# 文件路径: phase1/multimodel.py
import torch
import torch.nn as nn
from transformers import BertModel, Swinv2Model

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super(MultimodalModel, self).__init__()
        self.config = config
        
        # ================= Text Encoder (BERT) =================
        if config.use_text:
            self.text_model = BertModel.from_pretrained(config.text_model_name)
            # [Phase 1 Constraint] Freeze Backbone
            for param in self.text_model.parameters():
                param.requires_grad = False
            
            # Project BERT (768) -> text_dim (256)
            self.text_proj = nn.Linear(768, self.config.text_dim)
            
        # ================= Image Encoder (SwinV2) =================
        if config.use_image:
            self.image_model = Swinv2Model.from_pretrained(config.image_model_name)
            # [Phase 1 Constraint] Freeze Backbone
            for param in self.image_model.parameters():
                param.requires_grad = False
                
            # Project Swin (1024 for Base) -> image_dim (256 or 128)
            # 注意: SwinV2 Base 输出是 1024 维
            self.image_proj = nn.Linear(1024, self.config.image_dim)

        # ================= Classifier Head =================
        # 计算融合后的维度
        in_features = 0
        if config.use_text: in_features += config.text_dim
        if config.use_image: in_features += config.image_dim
        
        # [Phase 1 Constraint] Strong Classification Head
        # MLP: Fusion -> Linear -> LayerNorm -> GELU -> Dropout -> Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),       # 归一化加速收敛
            nn.GELU(),               # 更好的激活函数
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes)
        )

    def forward(self, text, image):
        features_list = []
        
        # 1. Text Flow
        if self.config.use_text:
            outputs = self.text_model(**text)
            # 使用 Pooler Output (CLS + Dense + Tanh)
            pooler_output = outputs.pooler_output 
            text_feat = self.text_proj(pooler_output)
            features_list.append(text_feat)

        # 2. Image Flow
        if self.config.use_image:
            outputs = self.image_model(image)
            # SwinV2 Pooler Output
            pooler_output = outputs.pooler_output
            img_feat = self.image_proj(pooler_output)
            features_list.append(img_feat)
            
        # 3. Fusion (Phase 1 Baseline: Concat)
        # 如果只有一个模态，直接使用；如果有两个，拼接
        if len(features_list) > 1:
            combined_features = torch.cat(features_list, dim=1)
        else:
            combined_features = features_list[0]
            
        # 4. Classification
        logits = self.classifier(combined_features)
        return logits