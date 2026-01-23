# 文件路径: phase1/utils/dataload.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class MultimodalDataset(Dataset):
    def __init__(self, data_dir, index_file, transform=None, is_train=True):
        """
        Args:
            data_dir: 存放图片和文本的具体目录 (../dataset/data)
            index_file: 索引文件路径 (../dataset/train_cleaned.txt)
            transform: 图像预处理 transforms
            is_train: 是否为训练模式 (测试集没有 label)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # 读取索引文件
        # train_cleaned.txt 格式: guid,tag
        self.df = pd.read_csv(index_file, dtype={'guid': str})
        
        # 标签映射
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        
        # 预先加载所有 guid 和 label 到内存列表
        self.data_list = []
        for _, row in self.df.iterrows():
            item = {'guid': str(row['guid']).strip()}
            if self.is_train and 'tag' in row and pd.notna(row['tag']):
                tag_str = str(row['tag']).strip()
                if tag_str in self.label_map:
                    item['label'] = self.label_map[tag_str]
                else:
                    continue # 跳过异常标签
            else:
                item['label'] = -1 # 测试集
            self.data_list.append(item)
            
        # 方便外部访问 label 分布
        if self.is_train:
            self.labels = [x['label'] for x in self.data_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        guid = item['guid']

        # === Phase 4 修改逻辑 ===
        # 如果是 VLM 增强样本 (例如 guid="123_vlm")
        # 文本读 "123_vlm.txt"，图片读 "123.jpg"
        if "_vlm" in guid:
            original_guid = guid.replace("_vlm", "")
            txt_path = os.path.join(self.data_dir, f"{guid}.txt") # 新生成的描述
            img_path_candidates = [
                os.path.join(self.data_dir, f"{original_guid}.jpg"),
                os.path.join(self.data_dir, f"{original_guid}.png")
            ]
        else:
            txt_path = os.path.join(self.data_dir, f"{guid}.txt")
            img_path_candidates = [
                os.path.join(self.data_dir, f"{guid}.jpg"),
                os.path.join(self.data_dir, f"{guid}.png")
            ]
        
        # 1. 加载文本
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        text_content = ""
        try:
            # Phase 0 已经统一转为 utf-8，这里直接读
            with open(txt_path, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
        except Exception:
            text_content = "" # 容错
            
        # 2. 加载图像
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_dir, f"{guid}.png")
            
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # 如果图片损坏，创建一个全黑图容错（虽然清洗过，但为了代码健壮性）
            image = Image.new('RGB', (256, 256), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # 3. 返回
        if self.is_train:
            return text_content, image, torch.tensor(item['label'], dtype=torch.long)
        else:
            return text_content, image, guid # 测试集返回 GUID 以便生成提交文件