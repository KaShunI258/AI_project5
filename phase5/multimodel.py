import torch
import torch.nn as nn
from transformers import BertModel, Swinv2Model

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super(MultimodalModel, self).__init__()
        self.config = config
        dim = config.text_dim
        
        # Frozen Backbones
        if config.use_text:
            self.text_model = BertModel.from_pretrained(config.text_model_name)
            for param in self.text_model.parameters(): param.requires_grad = False
            self.text_proj = nn.Linear(768, dim)
            self.text_ln = nn.LayerNorm(dim)
            
        if config.use_image:
            self.image_model = Swinv2Model.from_pretrained(config.image_model_name)
            for param in self.image_model.parameters(): param.requires_grad = False
            self.image_proj = nn.Linear(1024, dim)
            self.image_ln = nn.LayerNorm(dim)

        # Fusion: Attention
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True, dropout=config.dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim, config.num_classes)
        )

    def forward(self, text, image):
        # 1. 提取特征
        t_feat = self.text_ln(self.text_proj(self.text_model(**text).pooler_output))
        i_feat = self.image_ln(self.image_proj(self.image_model(image).pooler_output))
        
        # === Phase 5 核心：模态消融逻辑 ===
        # 通过 config.ablation_mode 动态控制特征置零
        ablation_mode = getattr(self.config, 'ablation_mode', 'none')
        
        if ablation_mode == 'text_only':
            # 屏蔽图像：将图像特征全置为 0
            i_feat = torch.zeros_like(i_feat)
            
        elif ablation_mode == 'image_only':
            # 屏蔽文本：将文本特征全置为 0
            t_feat = torch.zeros_like(t_feat)
        # ==================================
        
        # 2. Attention Fusion (Attention + Combine)
        raw_sum = t_feat + i_feat
        seq = torch.stack([t_feat, i_feat], dim=1)
        attn_out, _ = self.attention(seq, seq, seq)
        attn_pool = torch.mean(attn_out, dim=1)
        
        combined_features = torch.cat((raw_sum, attn_pool), dim=1)
        
        return self.classifier(combined_features)