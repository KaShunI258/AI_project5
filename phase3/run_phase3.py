import subprocess
import os
import pandas as pd

EXPERIMENTS = [
    # 1. 基线: Cross Entropy (参考基准)
    {'name': 'Exp1_Baseline_CE', 'loss_type': 'ce', 'use_sampler': 'False'},
    
    # 2. 策略A: ACB Loss (验证 H2: 解决难样本和边界问题)
    {'name': 'Exp2_ACB_Loss', 'loss_type': 'acb', 'use_sampler': 'False'},
    
    # 3. 策略B: Sampler (验证: 纯数据平衡是否有效)
    {'name': 'Exp3_Sampler', 'loss_type': 'ce', 'use_sampler': 'True'},
    
    # [新增] 4. 策略A + 策略B: ACB Loss + Sampler (验证: 双管齐下是否更强)
    {'name': 'Exp4_ACB_Plus_Sampler', 'loss_type': 'acb', 'use_sampler': 'True'},
]

def main():
    print("🚀 Starting Phase 3: Imbalance Handling (4 Experiments)...")
    os.makedirs("results", exist_ok=True)
    summary = []
    
    for exp in EXPERIMENTS:
        print(f"\n{'='*40}")
        print(f"Running {exp['name']}...")
        print(f"Config: Loss={exp['loss_type']}, Sampler={exp['use_sampler']}")
        print(f"{'='*40}")
        
        cmd = [
            'python', 'main.py',
            '--name', exp['name'],
            '--loss_type', exp['loss_type'],
            '--use_sampler', exp['use_sampler'],
            # 固定使用 Phase 2 搜索出的最优架构参数
            '--feature_fusion', 'attention_combine',
            '--text_dim', '256', 
            '--dropout', '0.1',
            '--learning_rate', '5e-5',
            '--num_epochs', '15'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # 读取结果并汇总
            history_path = f"results/{exp['name']}_history.csv"
            if os.path.exists(history_path):
                df = pd.read_csv(history_path)
                # 取 val_f1 最高的那个 epoch 的数据
                best_epoch = df.loc[df['val_f1'].idxmax()]
                summary.append({
                    'Experiment': exp['name'],
                    'Loss': exp['loss_type'],
                    'Sampler': exp['use_sampler'],
                    'Best_Val_F1': best_epoch['val_f1'],
                    'Best_Neutral_F1': best_epoch['neutral_f1']
                })
            else:
                print(f"⚠️ Warning: Result file not found for {exp['name']}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {exp['name']}: {e}")

    # 保存汇总表
    if summary:
        res_df = pd.DataFrame(summary)
        # 调整列顺序，好看一点
        cols = ['Experiment', 'Loss', 'Sampler', 'Best_Val_F1', 'Best_Neutral_F1']
        res_df = res_df[cols]
        
        res_df.to_csv("phase3_summary.csv", index=False)
        print("\n🏆 Phase 3 Complete! Summary:")
        print(res_df)
    else:
        print("\n❌ No results collected.")

if __name__ == "__main__":
    main()