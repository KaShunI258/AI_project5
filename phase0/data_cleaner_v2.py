import os
import pandas as pd
import re
import hashlib
from PIL import Image
from tqdm import tqdm
import chardet

# ================= 配置区域 =================
BASE_DIR = 'dataset'
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_LABEL_FILE = os.path.join(BASE_DIR, 'train.txt')
OUTPUT_CLEAN_FILE = os.path.join(BASE_DIR, 'train_cleaned.txt')
AUDIT_LOG_FILE = os.path.join(BASE_DIR, 'data_audit.csv')

# ================= 核心改进：感知哈希算法 =================

def dhash(image, hash_size=8):
    """
    计算图片的差异哈希 (Difference Hash)。
    即使文件二进制不同，只要图片长得一样，哈希值就一样。
    """
    # 1. 转为灰度图
    # 2. 缩放到 (hash_size + 1, hash_size)，比如 9x8
    # Use Image.Resampling.LANCZOS for better quality downsampling
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    
    pixels = list(image.getdata())
    
    diff = []
    # 3. 比较每行相邻像素
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = image.getpixel((col, row))
            pixel_right = image.getpixel((col + 1, row))
            # 左边比右边亮吗？
            diff.append(pixel_left > pixel_right)
            
    # 4. 转为十六进制字符串
    decimal_value = 0
    for index, value in enumerate(diff):
        if value:
            decimal_value += 2**index
            
    return hex(decimal_value)[2:]

# ================= 原有工具函数 (保持不变) =================

def clean_text_content(text_bytes):
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

    text = text.lstrip('\ufeff')
    text = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', ' <URL> ', text)
    text = re.sub(r'@[\w_]+', ' <USER> ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def check_image_validity(img_path):
    if not os.path.exists(img_path):
        return False, "Missing Image"
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True, "OK"
    except Exception:
        return False, "Corrupt Image"

def get_text_md5(text):
    """文本依然使用MD5，因为文本是离散符号，清洗后必须完全一致"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# ================= 主流程 =================

def main():
    print(f"Loading labels from {TRAIN_LABEL_FILE}...")
    try:
        df = pd.read_csv(TRAIN_LABEL_FILE, dtype={'guid': str})
    except Exception as e:
        print(f"Error reading train.txt: {e}")
        return

    removed_log = []
    # Key: dHash(img) + MD5(text) -> Value: List of records
    content_map = {} 

    print("Step 1: Integrity Check, Text Cleaning & Perceptual Hashing...")
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        guid = str(row['guid']).strip()
        label = str(row['tag']).strip()
        
        img_path = os.path.join(DATA_DIR, f"{guid}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(DATA_DIR, f"{guid}.png")
            
        txt_path = os.path.join(DATA_DIR, f"{guid}.txt")

        # 1. 图片处理
        # 注意：dHash 需要打开图片对象，而不仅仅是 verify
        is_valid = False
        img_dhash = None
        
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    # 计算感知哈希 (dHash)
                    # 这一步会找出 24.jpg 和 1403.jpg 的视觉相似性
                    img_dhash = dhash(img) 
                    is_valid = True
            except Exception as e:
                removed_log.append({'guid': guid, 'reason': 'Corrupt Image', 'label': label})
                continue
        else:
            removed_log.append({'guid': guid, 'reason': 'Missing Image', 'label': label})
            continue

        # 2. 文本处理
        if not os.path.exists(txt_path):
            removed_log.append({'guid': guid, 'reason': 'Missing Text File', 'label': label})
            continue
            
        try:
            with open(txt_path, 'rb') as f:
                raw_bytes = f.read()
            
            clean_text = clean_text_content(raw_bytes)
            if not clean_text: # Handle empty or None
                 removed_log.append({'guid': guid, 'reason': 'Empty/Bad Text', 'label': label})
                 continue

            # 覆写清洗后的文本 (UTF-8)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)
                
            txt_md5 = get_text_md5(clean_text)

        except Exception as e:
            removed_log.append({'guid': guid, 'reason': f'Text Error: {str(e)}', 'label': label})
            continue

        # 3. 组合键值 (视觉哈希 + 文本哈希)
        pair_key = f"{img_dhash}_{txt_md5}"
        
        if pair_key not in content_map:
            content_map[pair_key] = []
        content_map[pair_key].append({'guid': guid, 'label': label})

    print(f"\nStep 2: Conflict & Duplicate Detection (Visual + Semantic)...")
    
    final_guids = []
    
    for pair_key, items in content_map.items():
        unique_labels = set(item['label'] for item in items)
        
        if len(unique_labels) > 1:
            # 冲突发现！(Case 24 vs 1403)
            # 虽然 GUID 不同，文件二进制可能不同，但图文内容一致且标签冲突 -> 杀
            print(f"  [Conflict Detected] Visual/Text match but labels differ: {[i['guid'] for i in items]} -> Labels: {unique_labels}")
            for item in items:
                removed_log.append({
                    'guid': item['guid'], 
                    'reason': 'Label Conflict (Visual Match)', 
                    'label': item['label'],
                    'detail': f"Conflict with {[i['guid'] for i in items if i['guid'] != item['guid']]}"
                })
        else:
            # 重复数据
            final_guids.append([items[0]['guid'], items[0]['label']])
            if len(items) > 1:
                for drop_item in items[1:]:
                    removed_log.append({
                        'guid': drop_item['guid'], 
                        'reason': 'Duplicate Data (Visual Match)', 
                        'label': drop_item['label']
                    })

    # ================= 保存 =================
    
    print(f"\nStep 3: Saving results to {BASE_DIR}...")
    df_clean = pd.DataFrame(final_guids, columns=['guid', 'tag'])
    df_clean.to_csv(OUTPUT_CLEAN_FILE, index=False)
    
    if removed_log:
        df_audit = pd.DataFrame(removed_log)
        df_audit.to_csv(AUDIT_LOG_FILE, index=False)
        print(f"  - Cleaned Index: {OUTPUT_CLEAN_FILE} ({len(df_clean)} samples)")
        print(f"  - Audit Log:     {AUDIT_LOG_FILE} ({len(removed_log)} removed)")
    else:
        print("No bad data found.")

if __name__ == "__main__":
    confirm = input("Run updated visual-hash cleaning? (dataset/data txts will be overwritten) [yes/no]: ")
    if confirm.lower() == 'yes':
        main()