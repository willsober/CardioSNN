import torch
import psutil
import pynvml
import time
from threading import Thread
from queue import Queue
import pandas as pd
from pathlib import Path
import json

class ResourceMonitor:
    def __init__(self, log_dir='results/resource_usage'):
   
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.monitoring = False
        self.metrics_queue = Queue()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def start_monitoring(self, tag='default'):
      
        self.tag = tag
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
       
        self.monitoring = False
        self.monitor_thread.join()
        
        
        metrics = []
        while not self.metrics_queue.empty():
            metrics.append(self.metrics_queue.get())
       
        df = pd.DataFrame(metrics)
        df.to_csv(self.log_dir / f'{self.tag}_resource_usage.csv', index=False)
        
      
        stats = {
            'gpu_memory': {
                'mean': df['gpu_memory_used'].mean(),
                'max': df['gpu_memory_used'].max(),
                'min': df['gpu_memory_used'].min()
            },
            'gpu_utilization': {
                'mean': df['gpu_utilization'].mean(),
                'max': df['gpu_utilization'].max()
            },
            'memory_bandwidth': {
                'mean': df['memory_bandwidth'].mean()
            },
            'power_usage': {
                'mean': df['power_usage'].mean(),
                'max': df['power_usage'].max()
            },
            'duration_minutes': (time.time() - self.start_time) / 60
        }
        
       
        with open(self.log_dir / f'{self.tag}_resource_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
            
        return stats
        
    def _monitor_loop(self):
       
        while self.monitoring:
            try:
          
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                memory_used = info.used / 1024**2  # MB
                memory_total = info.total / 1024**2  # MB
                
           
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
                
              
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # W
                
              
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                
                metrics = {
                    'timestamp': time.time() - self.start_time,
                    'gpu_memory_used': memory_used,
                    'gpu_memory_total': memory_total,
                    'gpu_utilization': gpu_util,
                    'memory_bandwidth': memory_util,
                    'power_usage': power,
                    'cpu_percent': cpu_percent,
                    'ram_percent': ram_percent
                }
                
                self.metrics_queue.put(metrics)
                time.sleep(0.1)  
                
            except Exception as e:
                print(f"Error in monitoring: {e}")
                break

def format_resource_stats(stats):
    
    return {
        'GPU Memory Usage': f"{stats['gpu_memory']['mean']:.1f}MB (max: {stats['gpu_memory']['max']:.1f}MB)",
        'GPU Utilization': f"{stats['gpu_utilization']['mean']:.1f}% (max: {stats['gpu_utilization']['max']:.1f}%)",
        'Memory Bandwidth': f"{stats['memory_bandwidth']['mean']:.1f}%",
        'Power Usage': f"{stats['power_usage']['mean']:.1f}W (max: {stats['power_usage']['max']:.1f}W)",
        'Duration': f"{stats['duration_minutes']:.1f} minutes"
    }