import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_evaluation_report(results):
   
    os.makedirs('results', exist_ok=True)
    
    if 'noise_robustness' in results:
        plt.figure(figsize=(12, 6))
        noise_data = []
        
        for noise_type, noise_results in results['noise_robustness'].items():
            for level, metrics in noise_results.items():
                noise_data.append({
                    'Noise Type': noise_type,
                    'Noise Level': level,
                    'Accuracy': metrics['accuracy']
                })
        
        df_noise = pd.DataFrame(noise_data)
        sns.lineplot(data=df_noise, x='Noise Level', y='Accuracy', 
                    hue='Noise Type', marker='o')
        plt.title('Model Robustness Against Different Types of Noise')
        plt.grid(True)
        plt.savefig('results/noise_robustness.png')
        plt.close()
    
    if 'timing_performance' in results:
        plt.figure(figsize=(10, 5))
        batch_sizes = []
        throughputs = []
        latencies = []
        
        for batch_size, metrics in results['timing_performance'].items():
            batch_sizes.append(batch_size)
            throughputs.append(metrics['throughput'])
            latencies.append(metrics['mean_time'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Throughput plot
        ax1.plot(batch_sizes, throughputs, 'o-')
        ax1.set_title('Throughput vs Batch Size')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (samples/second)')
        ax1.grid(True)
        
        # Latency plot
        ax2.plot(batch_sizes, latencies, 'o-')
        ax2.set_title('Latency vs Batch Size')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Latency (ms)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/performance_metrics.png')
        plt.close()
    
    report = ["# Model Evaluation Report\n"]
    
    if 'noise_robustness' in results:
        report.append("## Noise Robustness Analysis")
        for noise_type, noise_results in results['noise_robustness'].items():
            report.append(f"\n### {noise_type.capitalize()} Noise")
            report.append("| Noise Level | Accuracy | F1 Score |")
            report.append("|-------------|----------|----------|")
            for level, metrics in noise_results.items():
                report.append(f"| {level:.2f} | {metrics['accuracy']:.4f} | {metrics['f1']:.4f} |")
    
    if 'timing_performance' in results:
        report.append("\n## Performance Analysis")
        report.append("\n### Batch Processing Performance")
        report.append("| Batch Size | Throughput (samples/s) | Latency (ms) |")
        report.append("|------------|----------------------|--------------|")
        for batch_size, metrics in results['timing_performance'].items():
            report.append(f"| {batch_size} | {metrics['throughput']:.2f} | {metrics['mean_time']:.2f} |")
    
    with open('results/evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("Evaluation report has been generated in 'results' directory")
    print("- Visualizations: noise_robustness.png, performance_metrics.png")
    print("- Report: evaluation_report.md")