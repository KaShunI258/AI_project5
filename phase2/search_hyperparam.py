import os
import itertools
import subprocess
import sys
import time
import re

# ================= 配置区域 =================
param_grid = {
    'feature_fusion': ['attention_combine'], 
    'text_dim': [256, 512],
    'learning_rate': [5e-5, 1e-4, 2e-4],
    'dropout': [0.1, 0.3, 0.5],
    'batch_size': [32]
}

SCRIPT_PATH = 'main.py' 
# ===========================================

def get_combinations(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))
    return keys, combinations

def parse_accuracy(output):
    match = re.search(r"Best validation accuracy:\s+([0-9.]+)", output)
    if match: return float(match.group(1))
    match_kfold = re.search(r"Average Validation Accuracy:\s+([0-9.]+)", output)
    if match_kfold: return float(match_kfold.group(1))
    return 0.0

def main():
    keys, combinations = get_combinations(param_grid)
    total_exps = len(combinations)
    
    print(f"🚀 开始超参数搜索 (Debug模式)，共 {total_exps} 组实验...")
    os.makedirs("search_logs", exist_ok=True)
    
    best_acc = 0.0
    best_config = None
    
    for idx, combo in enumerate(combinations):
        current_params = dict(zip(keys, combo))
        exp_name = f"search_exp_{idx+1}"
        
        print(f"\n[{idx+1}/{total_exps}] 正在运行: {current_params}")
        
        # 构建命令
        cmd = ['python', SCRIPT_PATH]
        for k, v in current_params.items():
            cmd.extend([f'--{k}', str(v)])
        
        # 固定参数
        cmd.extend([
            '--name', exp_name,
            '--num_epochs', '15',
            '--data_dir', '../dataset/data',
            '--train_file', '../dataset/train_cleaned.txt',
            '--test_file', '../dataset/test_without_label.txt'
            # 注意：移除了 --wandb False，因为 main.py 中 type=bool 会把 "False" 字符串解析为 True
            # 不传该参数则默认使用 main.py 中的 default=False
        ])
        
        try:
            # 运行并捕获输出
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # === 核心修改：如果失败，直接打印报错 ===
            if result.returncode != 0:
                print(f"❌ 实验崩溃 (Return Code: {result.returncode})")
                print("vvvvvvvvvv 错误信息 vvvvvvvvvv")
                print(result.stderr) # 打印完整报错
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                continue # 跳过当前组合
                
            # 解析结果
            acc = parse_accuracy(result.stdout)
            print(f"   -> 结果: Acc = {acc:.4f}")
            
            # 保存日志
            with open(os.path.join("search_logs", f"{exp_name}.log"), "w", encoding='utf-8') as f:
                f.write(result.stdout + "\n" + result.stderr)
            
            if acc > best_acc:
                best_acc = acc
                best_config = current_params
                print(f"   🔥 新的最佳结果! ({best_acc:.4f})")
                
        except Exception as e:
            print(f"   ❌ 脚本执行错误: {e}")
            
    print(f"\n🏆 最佳准确率: {best_acc:.4f}")
    print(f"🏆 最佳参数: {best_config}")

if __name__ == "__main__":
    main()