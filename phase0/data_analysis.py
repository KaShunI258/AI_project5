import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random

# ================= 配置 =================
BASE_DIR = 'dataset'
CLEAN_FILE = os.path.join(BASE_DIR, 'train_cleaned.txt')
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_SAVE_DIR = 'figures' # 图片保存目录

if not os.path.exists(FIG_SAVE_DIR):
    os.makedirs(FIG_SAVE_DIR)

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签，如果报错可改为 'Arial'
plt.rcParams['axes.unicode_minus'] = False 

def main():
    print(f"Loading cleaned data from {CLEAN_FILE}...")
    df = pd.read_csv(CLEAN_FILE, dtype={'guid': str})
    
    # ================= Fig 0-1: 类别分布 =================
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='tag', data=df, order=['negative', 'neutral', 'positive'], palette='viridis')
    plt.title('Fig 0-1: Label Distribution (Imbalance Check)', fontsize=14)
    plt.xlabel('Sentiment Label')
    plt.ylabel('Count')
    
    # 在柱子上标数值
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
        
    save_path = os.path.join(FIG_SAVE_DIR, 'Fig0-1_Label_Distribution.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

    # ================= Fig 0-2: 文本长度分布 =================
    print("Calculating text lengths...")
    text_lengths = []
    for guid in df['guid']:
        txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 简单按空格分词统计长度 (粗略估计)
                text_lengths.append(len(content.split()))
        except:
            pass

    plt.figure(figsize=(10, 6))
    sns.histplot(text_lengths, bins=50, kde=True, color='skyblue')
    plt.title('Fig 0-2: Text Length Distribution (Word Count)', fontsize=14)
    plt.xlabel('Length (words)')
    plt.ylabel('Frequency')
    plt.axvline(x=128, color='r', linestyle='--', label='len=128') # 辅助线
    plt.legend()
    
    save_path = os.path.join(FIG_SAVE_DIR, 'Fig0-2_Text_Length.png')
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

    # ================= Fig 0-3: Neutral 样本展示 =================
    print("Generating Neutral samples grid...")
    neutral_df = df[df['tag'] == 'neutral']
    
    # 随机抽 9 个
    if len(neutral_df) >= 9:
        samples = neutral_df.sample(9, random_state=42)
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        plt.suptitle('Fig 0-3: Random Neutral Samples', fontsize=16)
        
        for i, (idx, row) in enumerate(samples.iterrows()):
            guid = row['guid']
            ax = axes[i // 3, i % 3]
            
            # 读取图片
            img_path = os.path.join(DATA_DIR, f"{guid}.jpg")
            if not os.path.exists(img_path): img_path = os.path.join(DATA_DIR, f"{guid}.png")
            
            try:
                img = Image.open(img_path)
                ax.imshow(img)
            except:
                ax.text(0.5, 0.5, "Img Error", ha='center')
            
            # 读取文本 (取前30个字符)
            txt_path = os.path.join(DATA_DIR, f"{guid}.txt")
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text_snippet = f.read()[:30] + "..."
            except:
                text_snippet = "Txt Error"
                
            ax.set_title(f"ID:{guid}\n{text_snippet}", fontsize=9)
            ax.axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(FIG_SAVE_DIR, 'Fig0-3_Neutral_Samples.png')
        plt.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close()
    else:
        print("Not enough neutral samples to plot Fig 0-3.")

if __name__ == "__main__":
    main()