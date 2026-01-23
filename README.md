# 人工智能期末项目

这是一个基于数据为中心（Data-Centric AI）理念的多模态情感分析项目。本项目不仅仅依赖模型架构的改进，更通过引入多模态大模型（VLM）进行定向数据增强，成功解决了小样本多模态任务中常见的图文弱相关和类别不平衡问题。

## ✨ 核心亮点 (Highlights)

1.  **Data-Centric 策略**：不同于传统的噪声增强，本项目利用 **Qwen-VL** 生成客观图片描述，通过“大模型教小模型”的方式，显式地修复了原始数据中语义缺失的问题。
2.  **SOTA 级架构**：采用 **BERT (Text) + Swin Transformer V2 (Image)** 作为骨干网络，结合 **Attention Fusion** 机制进行特征融合。
3.  **不平衡处理**：引入 **ACB Loss (Adaptive Class Balancing Loss)**，动态调整权重，显著提升了 Neutral（难样本）类别的识别能力。
4.  **强基线对比**：在 Phase 6 中与 **CLIP (ViT-L/14)** 进行 Linear Probe 对比，证明了“全量微调+数据增强”的小模型在特定领域优于通用的预训练大模型。

---

## 📂 项目结构 (File Structure)

```text
AI_course_of_ECNU/
├── dataset/                     # 数据集根目录
│   ├── data_audit.csv           # 数据审计统计
│   ├── train_cleaned.txt        # 清洗后的训练集索引
│   ├── test_without_label.txt   # 测试集索引文件
│   └── data/                    # [核心数据] 存放所有 .jpg 图片和 .txt 文本
├── pretrained_models/           # [关键] 预训练模型文件夹
│   ├── bert-base-uncased/       # BERT 权重目录
│   └── swinv2-base-patch4-window8-256/ # SwinV2 权重目录
├── phase0/                      # 数据预处理阶段
│   ├── data_cleaner.py          # 数据清洗脚本
│   └── data_analysis.py         # 数据分布分析
├── phase1/                      # Baseline 搭建阶段
│   ├── main.py
│   ├── multimodel.py
│   ├── trainer.py
│   ├── visualization.py
│   ├── utils/                   # 工具包 (Config, Dataset)
│   └── results/                 # Phase 1 运行结果
├── phase2/                      # 架构搜索与超参优化
│   ├── run_phase2.py            # 搜索脚本
│   ├── search_hyperparam.py
│   ├── multimodel.py
│   ├── trainer.py
│   └── utils/
├── phase3/                      # Loss 函数探索阶段
│   ├── run_phase3.py            # 主运行脚本
│   ├── trainer.py               # 支持 ACB Loss 的训练器
│   ├── multimodel.py
│   ├── visualization_phase3.py
│   ├── utils/
│   └── results/                 # 存放 Exp2_ACB_Loss_best.pth 等
├── phase4/                      # [核心阶段] VLM 增强与最终模型
│   ├── run_phase4.py            # 全流程主控脚本
│   ├── augmentations.py         # VLM API 调用与增强逻辑
│   ├── multimodel.py            # 最终优化的模型架构
│   ├── trainer.py
│   ├── predict_test.py          # 测试集预测脚本
│   ├── visualization_phase4_final.py # 性能演进绘图
│   ├── utils/
│   ├── figures/                 # 存放 Fig4-1, 4-2, 4-3 等图表
│   └── results/                 # 存放 Phase4_Augmentation_best.pth (最优模型)
├── phase5/                      # 模态消融实验
│   ├── run_phase5.py            # 消融推理脚本
│   ├── visualization_phase5_radar.py # 雷达图绘制
│   ├── multimodel.py            # 支持 ablation_mode 的模型
│   ├── Tab5-1_Modality_Ablation.csv # 消融数据表
│   └── utils/
├── phase6/                      # CLIP 对比实验
│   ├── run_phase6.py            # CLIP 训练与对比主脚本
│   ├── clip_classifier.py       # CLIP Linear Probe 模型定义
│   ├── download_clip.py
│   ├── figures/                 # 存放对比图
│	├── pretrained_models/
│	│ └── clip-vit-large-patch14-336/     # CLIP 权重目录 (Phase 6)
│   └── utils/
├── predict.txt                  # 最终提交的测试集预测结果
├── requirements.txt             # 项目依赖库列表
└── README.md                    # 项目说明文档
```

------

## 🚀 快速开始

### 1. 环境安装

```Bash
conda create -n multimodel python=3.11
conda activate multimodel
pip install -r requirements.txt
```

### 2. 预训练模型准备

请从 Hugging Face 或其他渠道下载以下模型权重，并放置在 `pretrained_models/` 目录下：

- `bert-base-uncased`
- `microsoft/swinv2-base-patch4-window8-256`

此外，`openai/clip-vit-large-patch14-336` 仅 Phase 6 需要，因此需要放在`phase6/pretrained_models/`。

------

## 🏃‍♂️ 运行流程

本项目按 Phase 逐步推进，建议按顺序运行。

### Phase 1-3: 基础架构与 Loss 探索

这部分包含基础 Baseline 搭建、特征融合方式对比及 Loss 函数优化。

#### phase1

这是生成基线标准的命令行。

```bash
cd phase1
python main.py \
  --name Phase1_Baseline \
  --feature_fusion concat \
  --loss_type ce \
  --learning_rate 5e-5 \
  --dropout 0.1 \
  --num_epochs 15
```

#### phase2

执行融合方式探索：

```bash
cd phase2
# 该脚本会自动循环运行多次实验，并将结果汇总
python run_phase2.py
```

对最优融合方式进行超参数探索：

```bash
python search_hyperparam.py
```

#### phase3

```Bash
cd phase3
# 运行 Exp2 (使用 ACB Loss 训练最优 Baseline)
python main.py --name Exp2_ACB_Loss --loss_type acb --feature_fusion attention_combine
```

*产出*: `phase3/results/Exp2_ACB_Loss_best.pth` ，这将作为phase4的训练起点。

### Phase 4: VLM 定向增强与最终模型

这是本项目的核心阶段，包含难样本挖掘、VLM 增强和迁移学习。

1. **配置 API**: 打开 `phase4/run_phase4.py`，需要填入 ECNU Qwen-VL API Key。

2. **执行全流程**:

   ```Bash
   cd phase4
   python run_phase4.py
   ```

   *程序会自动执行：加载 Phase 3 模型 -> 挖掘难样本 -> 调用 API 增强 -> 生成新数据集 -> 重新微调。*

3. **可视化**:

   ```Bash
   python visualization_phase4_final.py
   ```

*产出*: `phase4/results/Phase4_Augmentation_best.pth` ，也就是最终最优模型。

### Phase 5: 模态消融实验

验证文本和图像模态各自的贡献。

```Bash
cd phase5
# 执行消融推理
python run_phase5.py
# 绘制高级雷达图
python visualization_phase5_radar.py
```

### Phase 6: CLIP Baseline 对比

对比我们的模型与 CLIP (Linear Probe) 的性能。

```Bash
cd phase6
# 训练 CLIP 并进行对比评估
python run_phase6.py
# 得到可视化结果
python visualization.py
```

------

## 📊 如何获得测试集结果

使用 Phase 4 训练出的最优模型对无标签测试集进行预测。

1. 确保你已经完成了 Phase 4 的训练，并且 `phase4/results/Phase4_Augmentation_best.pth` 存在。

2. 运行预测脚本：

   ```Bash
   cd phase4
   python predict_test.py
   ```
   
3. **结果文件**: 生成的文件位于 `phase4/predict.txt`。

   - 格式：`guid,tag`
   - 内容：包含所有测试集样本的预测标签（positive/neutral/negative）。

------

## 📚 参考资料

本项目代码实现参考了以下论文：

**论文 (Papers):**

1. **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019.
2. **Swin Transformer V2**: Liu et al., "Swin Transformer V2: Scaling Up Capacity and Resolution", CVPR 2022.
3. **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021.
4. **ACB Loss**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
5. **Qwen-VL**: Bai et al., "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond", 2023.

------
