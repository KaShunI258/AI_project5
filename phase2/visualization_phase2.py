import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置风格
sns.set(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_fusion_ladder():
    data_path = "phase2_summary.csv"
    if not os.path.exists(data_path):
        print("Run run_phase2.py first!")
        return
        
    df = pd.read_csv(data_path)
    
    # 排序：按照融合阶梯顺序
    order = ['concat', 'combine', 'attention', 'attention_concat', 'attention_combine', 'encoder']
    df['fusion_method'] = pd.Categorical(df['fusion_method'], categories=order, ordered=True)
    df = df.sort_values('fusion_method')
    
    # Fig 2-1: Overall Metric (Weighted F1)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='fusion_method', y='weighted_f1', data=df, palette='viridis')
    plt.title('Fig 2-1: Overall Performance vs. Fusion Method (Weighted F1)')
    plt.ylim(0.4, 0.8) # 根据实际数据调整
    plt.xlabel('Fusion Mechanism')
    plt.ylabel('Weighted F1 Score')
    
    # 添加数值标签
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
        
    plt.savefig('figures/Fig2-1_Fusion_Overall.png', dpi=300)
    plt.close()
    
    # Fig 2-2: Neutral Metric
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='fusion_method', y='neutral_f1', data=df, palette='magma')
    plt.title('Fig 2-2: Neutral Class Performance vs. Fusion Method')
    plt.xlabel('Fusion Mechanism')
    plt.ylabel('Neutral F1 Score')
    
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)
        
    plt.savefig('figures/Fig2-2_Fusion_Neutral.png', dpi=300)
    plt.close()
    
    print("Figures saved to phase2/figures/")

if __name__ == "__main__":
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plot_fusion_ladder()