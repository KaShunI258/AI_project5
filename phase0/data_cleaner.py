import os
import pandas as pd
import re
import hashlib
from PIL import Image
from tqdm import tqdm
import chardet

# ================= 配置区域 (已修正路径) =================
BASE_DIR = 'dataset'  # 根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')             # 图片和文本: dataset/data
TRAIN_LABEL_FILE = os.path.join(BASE_DIR, 'train.txt') # 原始标签: dataset/train.txt

# 输出文件也建议放在 dataset 目录下，保持整洁
OUTPUT_CLEAN_FILE = os.path.join(BASE_DIR, 'train_cleaned.txt') 
AUDIT_LOG_FILE = os.path.join(BASE_DIR, 'data_audit.csv')       

# ================= 工具函数 =================

def clean_text_content(text_bytes):
    """检测编码并转为 UTF-8，同时执行正则规范化"""
    # 1. 解码尝试
    try:
        text = text_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = text_bytes.decode('gb18030')
        except:
            result = chardet.detect(text_bytes)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            try:
                text = text_bytes.decode(encoding, errors='ignore')
            except:
                return None 

    # 2. 规范化清洗 (Source 32: 数据预处理)
    text = text.lstrip('\ufeff') # 去除 BOM
    # 替换 URL, @User
    text = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', ' <URL> ', text)
    text = re.sub(r'@[\w_]+', ' <USER> ', text)
    # 保留 #tag 中的文字
    text = re.sub(r'#(\w+)', r'\1', text)
    # 压缩空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def check_image_validity(img_path):
    """检查图片完整性"""
    if not os.path.exists(img_path):
        return False, "Missing Image"
    try:
        with Image.open(img_path) as img:
            img.verify() 
        return True, "OK"
    except Exception:
        return False, "Corrupt Image"

def get_md5(content):
    return hashlib.md5(content).hexdigest()

# ================= 主流程 =================

def main():
    print(f"Loading labels from {TRAIN_LABEL_FILE}...")
    
    # 读取原始标签 (Source 10)
    try:
        df = pd.read_csv(TRAIN_LABEL_FILE, dtype={'guid': str})
    except Exception as e:
        print(f"Error reading train.txt: {e}")
        return

    removed_log = []     
    content_map = {} # hash -> list of {guid, label}

    print("Step 1: Integrity Check & Text Cleaning...")
    # 遍历所有数据
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        guid = str(row['guid']).strip()
        label = str(row['tag']).strip()
        
        # 路径构建 (dataset/data/guid.jpg)
        img_path = os.path.join(DATA_DIR, f"{guid}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(DATA_DIR, f"{guid}.png") # 兼容 png
            
        txt_path = os.path.join(DATA_DIR, f"{guid}.txt")

        # --- 图片检查 ---
        is_img_valid, img_msg = check_image_validity(img_path)
        if not is_img_valid:
            removed_log.append({'guid': guid, 'reason': img_msg, 'label': label})
            continue

        # --- 文本检查与清洗 ---
        if not os.path.exists(txt_path):
            removed_log.append({'guid': guid, 'reason': 'Missing Text File', 'label': label})
            continue
            
        try:
            with open(txt_path, 'rb') as f:
                raw_bytes = f.read()
            
            if len(raw_bytes) == 0:
                removed_log.append({'guid': guid, 'reason': 'Empty Text', 'label': label})
                continue
                
            clean_text = clean_text_content(raw_bytes)
            if clean_text is None:
                removed_log.append({'guid': guid, 'reason': 'Text Encoding Error', 'label': label})
                continue
            
            # 覆写回 dataset/data/guid.txt (确保全是 UTF-8)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)

        except Exception as e:
            removed_log.append({'guid': guid, 'reason': f'Text Read Error: {str(e)}', 'label': label})
            continue

        # --- 计算 Hash 用于查重 ---
        with open(img_path, 'rb') as f:
            img_hash = get_md5(f.read())
        txt_hash = get_md5(clean_text.encode('utf-8'))
        
        pair_hash = img_hash + txt_hash
        
        if pair_hash not in content_map:
            content_map[pair_hash] = []
        content_map[pair_hash].append({'guid': guid, 'label': label})

    print(f"\nStep 2: Conflict & Duplicate Detection...")
    
    final_guids = []
    
    for pair_hash, items in content_map.items():
        unique_labels = set(item['label'] for item in items)
        
        if len(unique_labels) > 1:
            # 冲突：内容相同但标签不同 -> 全部剔除
            for item in items:
                removed_log.append({
                    'guid': item['guid'], 
                    'reason': 'Label Conflict', 
                    'label': item['label'],
                    'detail': 'Content match but label mismatch'
                })
        else:
            # 重复：保留第一个
            keep_item = items[0]
            final_guids.append([keep_item['guid'], keep_item['label']])
            
            if len(items) > 1:
                for drop_item in items[1:]:
                    removed_log.append({
                        'guid': drop_item['guid'], 
                        'reason': 'Duplicate Data', 
                        'label': drop_item['label']
                    })

    # ================= 保存结果 =================
    
    print(f"\nStep 3: Saving results to {BASE_DIR}...")
    df_clean = pd.DataFrame(final_guids, columns=['guid', 'tag'])
    df_clean.to_csv(OUTPUT_CLEAN_FILE, index=False)
    
    if removed_log:
        df_audit = pd.DataFrame(removed_log)
        df_audit.to_csv(AUDIT_LOG_FILE, index=False)
        print(f"  - Cleaned Index: {OUTPUT_CLEAN_FILE} ({len(df_clean)} samples)")
        print(f"  - Audit Log:     {AUDIT_LOG_FILE} ({len(removed_log)} removed)")
        print("\nRemoval Statistics:")
        print(df_audit['reason'].value_counts())
    else:
        print("No data needed cleaning.")

if __name__ == "__main__":
    confirm = input("This will overwrite txt files in dataset/data to UTF-8. Type 'yes' to proceed: ")
    if confirm.lower() == 'yes':
        main()