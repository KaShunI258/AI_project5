import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import BertTokenizer, AutoImageProcessor
from torchvision import transforms

# 引用本地模块
from utils.config import Config
from utils.dataload import MultimodalDataset  # <--- 用它来获取正确的标签映射
from multimodel import MultimodalModel

# === 配置路径 ===
MODEL_PATH = "results/Phase4_Augmentation_best.pth"
DATA_DIR = "../dataset/data"
TRAIN_FILE = "../dataset/train_cleaned.txt"       # 用来获取正确的 label map
TEST_FILE = "../dataset/test_without_label.txt"
OUTPUT_FILE = "predict.txt"

class Args:
    """模拟配置参数"""
    def __init__(self):
        self.text_model_name = "../pretrained_models/bert-base-uncased"
        self.image_model_name = "../pretrained_models/swinv2-base-patch4-window8-256"
        self.feature_fusion = 'attention_combine'
        self.text_dim = 256
        self.image_dim = 256
        self.num_classes = 3
        self.dropout = 0.1
        self.use_text = 1
        self.use_image = 1
        # 路径占位
        self.data_dir = DATA_DIR
        self.train_file = TRAIN_FILE
        self.test_file = "dummy"
        self.result_file = "dummy"
        # 其他参数
        self.batch_size = 1
        self.learning_rate = 1e-5
        self.num_epochs = 1
        self.val_ratio = 0.1
        self.early_stop_patience = 1
        self.loss_type = 'ce'
        self.use_sampler = False
        self.wandb = False
        self.name = "Inference"
        self.project_name = "Inference"
        self.log_iteration = 10

def read_text_file(guid):
    txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        return content
    # 如果没找到，打印警告
    # print(f"Warning: Text not found for {guid}")
    return ""

def read_image_file(guid):
    img_path = os.path.join(DATA_DIR, f"{guid}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(DATA_DIR, f"{guid}.png")
    
    if os.path.exists(img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except:
            pass
    # 打印警告，这是关键调试信息
    print(f"⚠️ Warning: Image not found for guid [{guid}]. Input will be black!")
    return Image.new('RGB', (224, 224), (0, 0, 0))

def main():
    print("🚀 Starting Prediction (Fixed Version)...")
    
    args = Args()
    config = Config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    # 1. 自动获取正确的 Label Map
    # 我们实例化一个临时 Dataset，利用它的逻辑来解析 label 顺序
    print("🔍 Detecting correct label mapping from training file...")
    temp_dataset = MultimodalDataset(config.data_dir, config.train_file, transform=None, is_train=True)
    
    # MultimodalDataset 通常会有 label_map 属性，或者我们根据 labels 推导
    # 假设 dataset.label_map 存在：{'negative': 0, 'neutral': 1, 'positive': 2}
    # 我们需要反转它：{0: 'negative', ...}
    if hasattr(temp_dataset, 'label_map'):
        label_map = temp_dataset.label_map
        id2label = {v: k for k, v in label_map.items()}
    else:
        # 如果没有直接属性，按字母排序重新生成一遍（这是 MultimodalDataset 的通用逻辑）
        unique_labels = sorted(list(set(temp_dataset.labels_str))) # 假设 labels_str 存了原始文本
        # 如果 dataset 没存 labels_str，我们直接硬编码最常见的逻辑：
        # 根据经验，Phase 代码通常使用 sorted list
        print("⚠️ dataset.label_map not found, using Sorted Default: ['negative', 'neutral', 'positive']")
        id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
    print(f"✅ Label Mapping: {id2label}")

    # 2. 加载模型
    tokenizer = BertTokenizer.from_pretrained(config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(config.image_model_name)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])

    model = MultimodalModel(config).to(device)
    if os.path.exists(MODEL_PATH):
        print(f"✅ Loading weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"❌ Model file not found: {MODEL_PATH}")
        return
    model.eval()

    # 3. 读取测试集
    df = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(df)} samples.")

    results = []

    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df)):
            # === [核心修复] 强制转换为整数再转字符串，去除 '.0' ===
            raw_guid = row['guid']
            try:
                # 兼容 float (8.0), int (8), string ("8.0")
                guid = str(int(float(raw_guid))) 
            except:
                guid = str(raw_guid)
            # =================================================
            
            # 数据读取
            text_content = read_text_file(guid)
            raw_image = read_image_file(guid)
            
            # 预处理
            encoded_text = tokenizer([text_content], padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)
            image_tensor = transform(raw_image).unsqueeze(0).to(device)
            
            # 推理
            outputs = model(encoded_text, image_tensor)
            pred_idx = torch.argmax(outputs, dim=1).item()
            
            # 映射
            pred_label = id2label[pred_idx]
            
            results.append({'guid': guid, 'tag': pred_label})

    # 4. 保存
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Prediction done! Saved to {OUTPUT_FILE}")
    print("Preview:")
    print(output_df.head())
    
    # 简单统计
    print("\nLabel Distribution:")
    print(output_df['tag'].value_counts())

if __name__ == "__main__":
    main()