import torch
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from src.models.snn import SNN, CombinedLoss
from src.data.dataset import create_dataloaders
from src.utils.training import train_model
from src.utils import ResourceMonitor, format_resource_stats  # 添加导入

def train_models():
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 加载数据
    with open('data/processed/processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    # 定义要训练的模型配置
    models_to_train = [
        {
            'name': 'baseline',
            'config': {
                'num_epochs': 100,
                'batch_size': 128,
                'lr': 0.001,
                'time_steps': 8
            }
        },
        {
            'name': 'larger_steps',
            'config': {
                'num_epochs': 100,
                'batch_size': 128,
                'lr': 0.001,
                'time_steps': 16
            }
        }
    ]
    
    # 创建结果目录
    os.makedirs('results/resource_usage', exist_ok=True)
    
    # 创建资源监控器
    monitor = ResourceMonitor()
    
    # 训练所有模型
    results = {}
    resource_stats = {}
    for model_config in models_to_train:
        name = model_config['name']
        print(f"\nTraining {name}...")
        
        # 开始监控
        monitor.start_monitoring(tag=name)
        
        # 训练模型
        model, model_results = train_model(
            processed_data,
            model_name=name,
            **model_config['config']
        )
        
        # 停止监控并获取统计数据
        stats = monitor.stop_monitoring()
        resource_stats[name] = stats
        results[name] = model_results
        
        # 打印资源使用情况
        print(f"\nResource usage for {name}:")
        for metric, value in format_resource_stats(stats).items():
            print(f"{metric}: {value}")
    
    # 创建资源使用对比表格
    resource_comparison = pd.DataFrame([
        {
            'Model': name,
            'Avg GPU Memory (MB)': stats['gpu_memory']['mean'],
            'Max GPU Util (%)': stats['gpu_utilization']['max'],
            'Avg Power (W)': stats['power_usage']['mean'],
            'Training Time (min)': stats['duration_minutes']
        }
        for name, stats in resource_stats.items()
    ])
    
    resource_comparison.to_csv('results/resource_comparison.csv', index=False)
    
    return results, resource_stats

def main():
    print("Starting model training and comparison...")
    
    # 训练模型并获取资源使用统计
    results, resource_stats = train_models()
    
    # 创建性能对比表格
    comparison_table = create_comparison_table(results)
    print("\nModel Performance Comparison:")
    print(comparison_table.to_string(index=False))
    
    # 打印资源使用对比
    print("\nResource Usage Comparison:")
    resource_df = pd.read_csv('results/resource_comparison.csv')
    print(resource_df.to_string(index=False))
    
    # 保存表格
    comparison_table.to_csv('results/model_comparison.csv', index=False)
    
    print("\nResults have been saved to the 'results' directory:")
    print("- Performance comparison: results/model_comparison.csv")
    print("- Resource usage: results/resource_comparison.csv")
    print("- Detailed resource logs: results/resource_usage/")

if __name__ == '__main__':
    main()