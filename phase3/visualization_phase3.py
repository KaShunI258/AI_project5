import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文标签(可选)
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = "results"
FIG_DIR = "figures"
if not os.path.exists(FIG_DIR): os.makedirs(FIG_DIR)

def plot_results():
    # 1. 读取汇总数据
    summary_path = "phase3_summary.csv"
    if not os.path.exists(summary_path): 
        print("Please run run_phase3.py first!")
        return
    df = pd.read_csv(summary_path)
    
    # 简优化实验名称显示
    name_map = {
        'Exp1_Baseline_CE': 'Baseline (CE)',
        'Exp2_ACB_Loss': 'ACB Loss',
        'Exp3_Sampler': 'Sampler Only',
        'Exp4_ACB_Plus_Sampler': 'ACB + Sampler'
    }
    df['ShortName'] = df['Experiment'].map(name_map)
    
    # ==========================================
    # Fig 3-1: 整体性能对比 (Weighted F1)
    # ==========================================
    plt.figure(figsize=(10, 6))
    # 使用渐变色区分不同策略强度
    ax = sns.barplot(x='ShortName', y='Best_Val_F1', data=df, palette='viridis')
    plt.title('Fig 3-1: Overall Performance Comparison (Weighted F1)', fontsize=14, pad=20)
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('Weighted F1 Score', fontsize=12)
    
    # 动态设置 Y 轴范围，让差异更明显
    y_min = df['Best_Val_F1'].min() - 0.05
    y_max = df['Best_Val_F1'].max() + 0.05
    plt.ylim(max(0, y_min), min(1, y_max))
    
    for i in ax.containers: ax.bar_label(i, fmt='%.4f', padding=3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/Fig3-1_Overall_F1.png", dpi=300)
    plt.close()

    # ==========================================
    # Fig 3-2: Neutral 类别提升 (核心指标)
    # ==========================================
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='ShortName', y='Best_Neutral_F1', data=df, palette='magma')
    plt.title('Fig 3-2: Neutral Class F1 Improvement (H2 Verification)', fontsize=14, pad=20)
    plt.xlabel('Strategy', fontsize=12)
    plt.ylabel('Neutral F1 Score', fontsize=12)
    
    y_min = df['Best_Neutral_F1'].min() - 0.05
    y_max = df['Best_Neutral_F1'].max() + 0.05
    plt.ylim(max(0, y_min), min(1, y_max))
    
    for i in ax.containers: ax.bar_label(i, fmt='%.4f', padding=3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/Fig3-2_Neutral_F1.png", dpi=300)
    plt.close()

    # ==========================================
    # Fig 3-3: Neutral 预测置信度分布 (KDE Plot)
    # 重点对比：Baseline vs ACB vs ACB+Sampler
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    # 选择要对比的实验 (Exp3 单独Sampler通常置信度不高，为了图表清晰可以不画，或者你全画也可以)
    target_exps = ['Exp1_Baseline_CE', 'Exp2_ACB_Loss', 'Exp4_ACB_Plus_Sampler']
    colors = {'Exp1_Baseline_CE': 'grey', 'Exp2_ACB_Loss': 'blue', 'Exp4_ACB_Plus_Sampler': 'red'}
    labels = {'Exp1_Baseline_CE': 'Baseline', 'Exp2_ACB_Loss': 'ACB Loss', 'Exp4_ACB_Plus_Sampler': 'ACB + Sampler'}
    
    has_data = False
    for exp_name in target_exps:
        pred_path = f"{RESULT_DIR}/{exp_name}_val_preds.csv"
        if not os.path.exists(pred_path): continue
        
        preds_df = pd.read_csv(pred_path)
        # 筛选出模型预测为 Neutral 的样本 (查看模型对其预测的信心)
        # 或者是筛选出真实标签为 Neutral 的样本 (查看模型对真 Neutral 的信心) <- 推荐后者用于分析难样本
        # 这里我们分析：所有【真实标签为 Neutral】的样本，模型给出的置信度分布
        neutral_samples = preds_df[preds_df['true_label'] == 1]
        
        if len(neutral_samples) > 0:
            sns.kdeplot(
                neutral_samples['confidence'], 
                label=labels[exp_name], 
                color=colors[exp_name],
                fill=True, 
                alpha=0.2,
                linewidth=2
            )
            has_data = True

    if has_data:
        plt.title('Fig 3-3: Confidence Distribution on Neutral Samples', fontsize=14)
        plt.xlabel('Prediction Probability (Confidence)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.xlim(0, 1.0)
        plt.legend(title='Experiment')
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/Fig3-3_Confidence_KDE.png", dpi=300)
        print("Figures saved to phase3/figures/")
    else:
        print("No prediction data found for KDE plot.")
    
    plt.close()

if __name__ == "__main__":
    plot_results()