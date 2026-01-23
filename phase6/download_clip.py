from huggingface_hub import snapshot_download
import os

# 指定下载目录
local_dir = "./pretrained_models/clip-vit-large-patch14-336"
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading CLIP ViT-L/14@336px to {local_dir}...")

# 下载模型 (OpenAI 官方权重)
snapshot_download(
    repo_id="openai/clip-vit-large-patch14-336",
    local_dir=local_dir,
    ignore_patterns=["*.msgpack", "*.h5", "*.tflite"] # 只下载 PyTorch 权重
)

print("Download complete! Please upload the 'pretrained_models' folder to your server.")