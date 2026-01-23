import torch
import torch.nn as nn
from transformers import BertModel, Swinv2Model

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super(MultimodalModel, self).__init__()
        self.config = config
        
        # 维度配置：建议在 run_phase2.py 里把 dim 改为 512
        dim = config.text_dim # 假设 text_dim == image_dim
        
        # ================= Frozen Backbones =================
        if config.use_text:
            self.text_model = BertModel.from_pretrained(config.text_model_name)
            for param in self.text_model.parameters():
                param.requires_grad = False
            self.text_proj = nn.Linear(768, dim)
            self.text_ln = nn.LayerNorm(dim)  # [优化] 增加 LayerNorm
            
        if config.use_image:
            self.image_model = Swinv2Model.from_pretrained(config.image_model_name)
            for param in self.image_model.parameters():
                param.requires_grad = False
            self.image_proj = nn.Linear(1024, dim)
            self.image_ln = nn.LayerNorm(dim) # [优化] 增加 LayerNorm

        # ================= Fusion Modules =================
        # 使用较小的 Dropout 防止破坏特征
        fusion_dropout = 0.1 
        
        if 'attention' in config.feature_fusion or 'encoder' in config.feature_fusion:
            self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True, dropout=fusion_dropout)
        
        if config.feature_fusion == 'encoder':
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True, dropout=fusion_dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ================= Classifier Head =================
        if config.feature_fusion == 'concat':
            in_features = dim * 2 
        elif config.feature_fusion == 'combine':
            in_features = dim       
        elif config.feature_fusion == 'attention':
            in_features = dim
        elif config.feature_fusion == 'attention_concat':
            in_features = dim * 2 + dim  # 3 * dim
        elif config.feature_fusion == 'attention_combine':
            in_features = dim * 2        # (t+v) + attn -> dim + dim
        elif config.feature_fusion == 'encoder':
            in_features = dim
        else:
            raise ValueError(f"Unknown fusion method: {config.feature_fusion}")

        self.classifier = nn.Sequential(
            nn.Linear(in_features, dim), # 中间层保持宽一点
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout),  # 分类头的 dropout 可以保持 0.3
            nn.Linear(dim, config.num_classes)
        )

    def forward(self, text, image):
        # 1. Feature Extraction & Projection & Normalization
        # 加入 LayerNorm 是让 Attention 奏效的关键！
        t_raw = self.text_proj(self.text_model(**text).pooler_output)
        t_feat = self.text_ln(t_raw) 
        
        i_raw = self.image_proj(self.image_model(image).pooler_output)
        i_feat = self.image_ln(i_raw)
        
        fusion_type = self.config.feature_fusion
        combined_features = None

        # 2. Fusion Logic
        if fusion_type == 'concat':
            combined_features = torch.cat((t_feat, i_feat), dim=1)
            
        elif fusion_type == 'combine':
            combined_features = t_feat + i_feat
            
        elif fusion_type == 'attention':
            seq = torch.stack([t_feat, i_feat], dim=1) 
            attn_out, _ = self.attention(seq, seq, seq)
            combined_features = torch.mean(attn_out, dim=1)
            
        elif fusion_type == 'attention_concat':
            # H1: 原始信息 + 交互信息
            raw_concat = torch.cat((t_feat, i_feat), dim=1)
            
            seq = torch.stack([t_feat, i_feat], dim=1)
            attn_out, _ = self.attention(seq, seq, seq)
            attn_pool = torch.mean(attn_out, dim=1)
            
            combined_features = torch.cat((raw_concat, attn_pool), dim=1)
            
        elif fusion_type == 'attention_combine':
            # 使用 cat 而不是 +
            raw_sum = t_feat + i_feat
            
            seq = torch.stack([t_feat, i_feat], dim=1)
            attn_out, _ = self.attention(seq, seq, seq)
            attn_pool = torch.mean(attn_out, dim=1)
            
            # 这里必须拼接，维度才是 2*dim
            combined_features = torch.cat((raw_sum, attn_pool), dim=1)
            
        elif fusion_type == 'encoder':
            seq = torch.stack([t_feat, i_feat], dim=1)
            enc_out = self.transformer_encoder(seq)
            combined_features = torch.mean(enc_out, dim=1)

        # 3. Classification
        logits = self.classifier(combined_features)
        return logits