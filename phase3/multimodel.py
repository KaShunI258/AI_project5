import torch
import torch.nn as nn
from transformers import BertModel, Swinv2Model

class MultimodalModel(nn.Module):
    def __init__(self, config):
        super(MultimodalModel, self).__init__()
        self.config = config
        dim = config.text_dim # 推荐 512
        
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
        
        # Classifier (Attention+Combine -> dim*2)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim, config.num_classes)
        )

    def forward(self, text, image):
        t_feat = self.text_ln(self.text_proj(self.text_model(**text).pooler_output))
        i_feat = self.image_ln(self.image_proj(self.image_model(image).pooler_output))
        
        # Attention + Combine Logic
        raw_sum = t_feat + i_feat
        seq = torch.stack([t_feat, i_feat], dim=1)
        attn_out, _ = self.attention(seq, seq, seq)
        attn_pool = torch.mean(attn_out, dim=1)
        
        combined_features = torch.cat((raw_sum, attn_pool), dim=1) # 512 + 512 = 1024
        
        return self.classifier(combined_features)