import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# === 配置 ===
CSV_PATH = "results/Tab5-1_Modality_Ablation.csv" # 确保这个路径对
SAVE_DIR = "figures"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# 设置风格 (使用 seaborn 的高级配色和网格风格)
sns.set(style="whitegrid", font_scale=1.2)
# 解决特定环境下中文字体显示问题(虽然这里用英文)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def plot_advanced_radar_chart():
    print("Generating Advanced Radar Chart for Phase 5...")
    
    # 1. 读取数据
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: Result file {CSV_PATH} not found. Please run run_phase5.py first.")
        return
    df = pd.read_csv(CSV_PATH)

    # 2. 定义要展示的指标维度 (确保这些列在CSV里存在)
    # 我们选择最重要的几个指标
    categories = ['Accuracy', 'Weighted F1', 'Neutral F1', 'Precision', 'Recall']
    N = len(categories)

    # 3. 计算角度
    # 我们需要将圆周平分成 N 份
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # 为了让图表闭合，需要把第一个角度重复添加到末尾
    angles += angles[:1]

    # 4. 初始化极坐标图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 设置整体旋转角度，让第一个指标(Accuracy)在正上方
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1) # 顺时针方向

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='black', size=12)

    # Draw ylabels (径向刻度网格)
    ax.set_rlabel_position(0) # 将径向标签移到特定角度，避免挡住图形
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.0) # 固定 Y 轴范围在 0-1 之间

    # 5. 定义不同模态的数据和样式
    # 使用 seaborn 的深色调色板，显得专业
    palette = sns.color_palette("deep", n_colors=3)
    modalities_config = [
        # 确保这里的 Modality 名称与 CSV 中的完全一致
        {"name": "Text + Image (Full)", "color": palette[0], "marker": 'o'},
        {"name": "Text Only",          "color": palette[1], "marker": 's'},
        {"name": "Image Only",         "color": palette[2], "marker": '^'}
    ]

    # 6. 循环绘制每一条线
    for config in modalities_config:
        modality_name = config["name"]
        try:
            # 获取该模态对应指标的数值
            values = df[df['Modality'] == modality_name][categories].values.flatten().tolist()
            # 为了闭合，重复第一个值
            values += values[:1]
            
            # 绘制线条
            ax.plot(angles, values, color=config["color"], linewidth=2, linestyle='solid', label=modality_name, marker=config["marker"], markersize=8)
            # 填充颜色 (半透明)
            ax.fill(angles, values, color=config["color"], alpha=0.25)
            
        except IndexError:
            print(f"⚠️ Warning: Modality '{modality_name}' not found in CSV data. Skipping.")
            continue

    # 7. 添加图例和标题
    # 将图例放置在图表外侧右上角，避免遮挡
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Modality Settings", fontsize=12, title_fontsize=14)
    
    plt.title("Fig 5-2: Modality Capability Radar (Ablation Study)", size=18, color='black', y=1.08)

    # 调整布局确保不溢出
    plt.tight_layout()

    # 8. 保存
    save_path = f"{SAVE_DIR}/Fig5-2_Modality_Radar.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # bbox_inches='tight' 确保图例被完整保存
    plt.close()
    print(f"✅ Advanced radar chart saved to {save_path}")

if __name__ == "__main__":
    plot_advanced_radar_chart()