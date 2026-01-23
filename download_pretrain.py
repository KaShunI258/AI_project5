import os
from transformers import BertModel, BertTokenizer, Swinv2Model, AutoImageProcessor

# 创建目录
os.makedirs("./pretrained_models/bert-base-uncased", exist_ok=True)
os.makedirs("./pretrained_models/swinv2-base-patch4-window8-256", exist_ok=True)

print("正在下载 BERT...")
# 下载 BERT-Base (Uncased)
bert_name = "bert-base-uncased"
BertModel.from_pretrained(bert_name).save_pretrained("./pretrained_models/bert-base-uncased")
BertTokenizer.from_pretrained(bert_name).save_pretrained("./pretrained_models/bert-base-uncased")

print("正在下载 Swin Transformer V2...")
# 下载 SwinV2-Base (输入尺寸 256x256)
# 注意：你需要安装 timm 库: pip install timm
swin_name = "microsoft/swinv2-base-patch4-window8-256"
Swinv2Model.from_pretrained(swin_name).save_pretrained("./pretrained_models/swinv2-base-patch4-window8-256")
AutoImageProcessor.from_pretrained(swin_name).save_pretrained("./pretrained_models/swinv2-base-patch4-window8-256")

print("下载完成！")