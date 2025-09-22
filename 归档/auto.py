import os
import torch
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime
from itertools import product
from multiprocessing import Pool
import time

# 实验命令模板
EXPERIMENT_COMMAND_TEMPLATE = """
python main.py --is_transfer True --test_dataset {test_dataset} --pretrained_model_name {pretrained_model_name} --few {few} --shot {shot}
"""

# 可用的预训练模型
PRETRAINED_MODELS = {
    'PubMed': '/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/PubMed.GRACE.GAT.hyp_True.True.20250912-232538.pth',
    'CiteSeer': '/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/CiteSeer.GRACE.GAT.hyp_True.True.20250912-232342.pth',
    'Cora': '/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/Cora.GRACE.GAT.hyp_True.True.20250912-232536.pth',
    'Photo': '/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/Photo.GRACE.GAT.hyp_True.True.20250912-232537.pth',
    'Computers': '/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/Computers.GRACE.GAT.hyp_True.True.20250912-232538.pth',
}


# 获取预训练模型对应的字段信息
def get_pretrained_model_info(pretrained_model_path):
    basename = os.path.basename(pretrained_model_path)
    parts = basename.split('.')
    dataset = parts[0]  # 例如 "Cora"
    return dataset

# 保存实验日志
def save_experiment_log(experiment_id, log_data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_filename = f"experiment_log_{timestamp}.txt"
    
    with open(log_filename, "a") as f:
        f.write(f"Experiment {experiment_id} - {timestamp}:\n")
        for line in log_data:
            f.write(line + "\n")
        f.write("="*50 + "\n")

# 运行实验命令并返回最大测试准确率
def run_experiment(command, experiment_id):
    start_time = time.time()
    log_data = []
    
    # 执行命令并输出实验日志
    log_data.append(f"Running experiment {experiment_id}...")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    
    # 输出并记录实验的标准输出和错误信息
    log_data.append(f"Experiment {experiment_id} output:\n{out.decode('utf-8')}")
    if err:
        log_data.append(f"Experiment {experiment_id} error:\n{err.decode('utf-8')}")
    
    # 记录日志文件
    save_experiment_log(experiment_id, log_data)
    
    # 从日志中解析出 test 的准确率
    try:
        for line in out.decode('utf-8').splitlines():
            if 'Best @ epoch' in line:
                # 解析出 'val' 和 'test' 准确率
                parts = line.split(":")
                val_acc = float(parts[1].split(",")[0].split('=')[1].strip())  # 解析 val
                test_acc = float(parts[1].split(",")[1].split('=')[1].strip())  # 解析 test
                
                print(f"Experiment {experiment_id} finished: val_acc = {val_acc}, test_acc = {test_acc}")
                
                # 记录测试集准确率
                end_time = time.time()
                run_duration = end_time - start_time
                log_data.append(f"Experiment {experiment_id} completed in {run_duration:.2f} seconds.")
                save_experiment_log(experiment_id, log_data)
                
                return test_acc  # 返回 test 的准确率
        # 如果没有找到 max_test_acc，可以返回一个默认值（例如0）
        print(f"Experiment {experiment_id} finished: no test_acc found.")
        return 0.0
    except Exception as e:
        print(f"Error parsing output in experiment {experiment_id}: {e}")
        end_time = time.time()
        run_duration = end_time - start_time
        log_data.append(f"Experiment {experiment_id} completed in {run_duration:.2f} seconds.")
        save_experiment_log(experiment_id, log_data)
        return 0.0  # 如果发生异常，返回0.0作为默认值
# 计算每组实验的平均值和标准差
def run_single_experiment(test_dataset, pretrained_model_name, few, shot):
    experiment_results = []
    for i in range(5):  # 每个实验重复5次
        print(f"Running {i+1}/5 for {test_dataset} with pretrained model {pretrained_model_name} (Few: {few}, Shot: {shot})")
        command = EXPERIMENT_COMMAND_TEMPLATE.format(
            test_dataset=test_dataset, 
            pretrained_model_name=pretrained_model_name, 
            few=few, 
            shot=shot
        )
        result = run_experiment(command, f"{test_dataset}-{pretrained_model_name}-{few}-{shot}-{i+1}")
        
        # 如果 result 为 None，则跳过
        if result is not None:
            experiment_results.append(result)
    
    # 如果 experiment_results 为空，避免 np.mean 和 np.std 错误
    if len(experiment_results) > 0:
        return np.mean(experiment_results), np.std(experiment_results)
    else:
        return 0.0, 0.0  # 如果没有有效的实验结果，返回默认值

# 保存所有实验结果
def save_experiment_results(results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = f"experiment_results_{timestamp}.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"实验结果已保存到 {output_path}")

# 并行化实验函数，提升到全局作用域
def parallel_experiment(pretrained_model_path, results):
    # 获取预训练模型信息
    dataset = get_pretrained_model_info(pretrained_model_path)
    
    # 定义需要执行的实验配置
    for test_dataset, few, shot in product(PRETRAINED_MODELS.keys(), [False, True], [5, 10]):
        few_shot_str = f'{shot}shot' if few else 'public'
        
        # 运行该配置的实验
        mean_acc, std_acc = run_single_experiment(test_dataset, pretrained_model_path, few, shot)
        
        # 收集实验结果
        results.append({
            'Pretrained Model': pretrained_model_path,
            'Test Dataset': test_dataset,
            'Few Shot': few_shot_str,
            'Mean Accuracy': mean_acc,
            'Std Accuracy': std_acc,
        })

# 自动化实验函数
def run_all_experiments(pretrained_model_paths=None):
    if pretrained_model_paths is None:
        pretrained_model_paths = PRETRAINED_MODELS.values()
    
    # 保存实验数据的列表
    results = []
    
    # 使用 Pool 来并行化执行实验
    with Pool(processes=5) as pool:
        pool.starmap(parallel_experiment, [(path, results) for path in pretrained_model_paths])
    
    # 保存实验结果到文件
    save_experiment_results(results)

# 只执行某些特定预训练模型的实验
def run_partial_experiment(pretrained_model_path):
    run_all_experiments([pretrained_model_path])

# 主函数示例
if __name__ == "__main__":
    # 如果需要全量实验，调用 run_all_experiments()
    # run_all_experiments()  # 自动执行所有预训练模型的实验
    
    # 如果只需执行指定的某个实验，调用 run_partial_experiment
    # 例如执行 Computer 数据集的实验：
    run_partial_experiment("/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/Photo.GRACE.GAT.hyp_True.True.20250912-232537.pth")
    run_partial_experiment("/mnt/data1/Graph/HypGraphLoRA/pre_trained_gnn/Computers.GRACE.GAT.hyp_True.True.20250912-232538.pth")
