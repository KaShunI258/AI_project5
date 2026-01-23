import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import random
import os
import base64
from openai import OpenAI  # 需要安装: pip install openai

# 如果没有安装 nlpaug，提供一个空的占位符以防报错
try:
    import nlpaug.augmenter.word as naw
except ImportError:
    print("Warning: 'nlpaug' not found. Text augmentation will be disabled.")
    naw = None

class DataAugmenter:
    """
    基础数据增强类 (同义词替换 + 图像扰动)
    """
    def __init__(self):
        # 1. 文本基础增强 (同义词替换)
        self.text_aug = None
        if naw is not None:
            try:
                # 尝试初始化 wordnet，如果下载失败则跳过
                self.text_aug = naw.SynonymAug(aug_src='wordnet')
            except Exception as e:
                print(f"Text Aug init failed: {e}")

        # 2. 图像基础增强 (弱增强，防止破坏语义)
        self.basic_img_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=10),
        ])

    def basic_augment(self, text, image):
        """基础增强：随机对文本或图像进行扰动"""
        aug_text = text
        aug_image = image
        
        # 50% 概率增强文本
        if self.text_aug and random.random() > 0.5:
            try:
                # nlpaug 返回的是 list
                res = self.text_aug.augment(text)
                if isinstance(res, list):
                    aug_text = res[0]
                else:
                    aug_text = res
            except:
                pass
                
        # 50% 概率增强图像
        if random.random() > 0.5:
            aug_image = self.basic_img_transform(image)
            
        return aug_text, aug_image

class VLMAugmenter:
    """
    [Phase 4 核心] VLM API 调用接口
    使用 ECNU API 为 Bad Case 生成图片描述
    """
    def __init__(self, 
                 api_key="sk-ee3a6bcdb0e442be9259d84599b03675", 
                 base_url="https://chat.ecnu.edu.cn/open/api/v1", 
                 model="ecnu-vl"):
        
        self.model = model
        # 初始化 OpenAI 客户端 (适配 ECNU 接口)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        print(f"✅ VLM Augmenter initialized with model: {self.model}")

    def encode_image(self, image_path):
        """将本地图片转为 Base64 格式"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_caption(self, image_path):
        """
        调用 ECNU 多模态大模型生成图片描述
        """
        try:
            # 1. 编码图片
            base64_image = self.encode_image(image_path)
            
            # 2. 发送请求
            # Prompt 设计重点：要求客观描述，避免主观情感 (Label Leakage)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Please provide a detailed and objective description of the visual content of this image. Do not analyze the sentiment or emotion."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300  # 限制输出长度
            )
            
            caption = response.choices[0].message.content
            return caption.strip()
            
        except Exception as e:
            print(f"❌ API Error processing {image_path}: {e}")
            # 失败返回空字符串，后续逻辑会跳过此样本
            return ""

    def augment_dataset(self, bad_case_guids, data_dir, output_file):
        """
        核心流程：遍历 Bad Cases -> 生成描述 -> 保存新样本索引
        """
        print(f"🚀 Starting VLM Augmentation for {len(bad_case_guids)} bad cases...")
        
        # 1. 读取原始索引文件 (train_cleaned.txt)
        # 假设在上两级目录 ../dataset/train_cleaned.txt
        original_index_path = os.path.join(data_dir, "..", "train_cleaned.txt")
        if not os.path.exists(original_index_path):
            print(f"Error: Original index file not found at {original_index_path}")
            return

        with open(original_index_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        header = lines[0]
        # 建立 guid -> raw_line 映射
        data_map = {}
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                data_map[parts[0]] = line.strip()
        
        new_sample_lines = []
        success_count = 0
        
        # 2. 遍历 Bad Cases
        for i, guid in enumerate(bad_case_guids):
            if guid not in data_map: continue
            
            # 打印进度
            if i % 5 == 0:
                print(f"   Processing {i}/{len(bad_case_guids)}: {guid} ...")

            original_line = data_map[guid]
            _, label = original_line.split(',')
            
            # 寻找图片 (jpg 或 png)
            img_path = os.path.join(data_dir, f"{guid}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(data_dir, f"{guid}.png")
            
            if not os.path.exists(img_path):
                print(f"   Skip {guid}: Image not found.")
                continue
            
            # === 调用 API ===
            new_text = self.generate_caption(img_path)
            
            if not new_text:
                continue # API 失败则跳过
            
            # === 保存新样本 ===
            # 新的 guid 命名为 {guid}_vlm
            new_guid = f"{guid}_vlm"
            
            # 写入新的文本文件
            new_txt_path = os.path.join(data_dir, f"{new_guid}.txt")
            with open(new_txt_path, 'w', encoding='utf-8') as f:
                f.write(new_text)
            
            # 记录到索引列表 (图片复用逻辑由 Dataset 类处理)
            new_sample_lines.append(f"{new_guid},{label}\n")
            success_count += 1
            
        # 3. 合并保存：原始数据 + 新增数据
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write(header)
            # 写入原始数据
            f.writelines(lines[1:])
            # 写入新增数据
            f.writelines(new_sample_lines)
            
        print(f"\n✅ Augmentation Complete!")
        print(f"   - Original samples: {len(lines)-1}")
        print(f"   - VLM Augmented samples: {success_count}")
        print(f"   - Total training samples: {len(lines)-1 + success_count}")
        print(f"   - Saved to: {output_file}")