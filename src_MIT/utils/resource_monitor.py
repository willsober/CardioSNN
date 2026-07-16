# import torch
# import psutil
# import pynvml
# import time
# from threading import Thread
# from queue import Queue
# import pandas as pd
# from pathlib import Path
# import json
#
# class ResourceMonitor:
#     def __init__(self, log_dir='results/resource_usage'):
#
#         pynvml.nvmlInit()
#         self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#         self.monitoring = False
#         self.metrics_queue = Queue()
#         self.log_dir = Path(log_dir)
#         self.log_dir.mkdir(parents=True, exist_ok=True)
#
#     def start_monitoring(self, tag='default'):
#
#         self.tag = tag
#         self.monitoring = True
#         self.start_time = time.time()
#         self.monitor_thread = Thread(target=self._monitor_loop)
#         self.monitor_thread.start()
#
#     def stop_monitoring(self):
#
#         self.monitoring = False
#         self.monitor_thread.join()
#
#
#         metrics = []
#         while not self.metrics_queue.empty():
#             metrics.append(self.metrics_queue.get())
#
#         df = pd.DataFrame(metrics)
#         df.to_csv(self.log_dir / f'{self.tag}_resource_usage.csv', index=False)
#
#
#         stats = {
#             'gpu_memory': {
#                 'mean': df['gpu_memory_used'].mean(),
#                 'max': df['gpu_memory_used'].max(),
#                 'min': df['gpu_memory_used'].min()
#             },
#             'gpu_utilization': {
#                 'mean': df['gpu_utilization'].mean(),
#                 'max': df['gpu_utilization'].max()
#             },
#             'memory_bandwidth': {
#                 'mean': df['memory_bandwidth'].mean()
#             },
#             'power_usage': {
#                 'mean': df['power_usage'].mean(),
#                 'max': df['power_usage'].max()
#             },
#             'duration_minutes': (time.time() - self.start_time) / 60
#         }
#
#         stats = self._convert_int64_to_int(stats)
#
#         with open(self.log_dir / f'{self.tag}_resource_stats.json', 'w') as f:
#             json.dump(stats, f, indent=4)
#
#         return stats
#
#     def _monitor_loop(self):
#
#         while self.monitoring:
#             try:
#
#                 info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
#                 memory_used = info.used / 1024**2  # MB
#                 memory_total = info.total / 1024**2  # MB
#
#
#                 utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
#                 gpu_util = utilization.gpu
#                 memory_util = utilization.memory
#
#
#                 power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # W
#
#
#                 cpu_percent = psutil.cpu_percent()
#                 ram_percent = psutil.virtual_memory().percent
#
#                 metrics = {
#                     'timestamp': time.time() - self.start_time,
#                     'gpu_memory_used': memory_used,
#                     'gpu_memory_total': memory_total,
#                     'gpu_utilization': gpu_util,
#                     'memory_bandwidth': memory_util,
#                     'power_usage': power,
#                     'cpu_percent': cpu_percent,
#                     'ram_percent': ram_percent
#                 }
#
#                 self.metrics_queue.put(metrics)
#                 time.sleep(0.1)
#
#             except Exception as e:
#                 print(f"Error in monitoring: {e}")
#                 break
#
def format_resource_stats(stats):

    return {
        'GPU Memory Usage': f"{stats['gpu_memory']['mean']:.1f}MB (max: {stats['gpu_memory']['max']:.1f}MB)",
        'GPU Utilization': f"{stats['gpu_utilization']['mean']:.1f}% (max: {stats['gpu_utilization']['max']:.1f}%)",
        'Memory Bandwidth': f"{stats['memory_bandwidth']['mean']:.1f}%",
        'Power Usage': f"{stats['power_usage']['mean']:.1f}W (max: {stats['power_usage']['max']:.1f}W)",
        'Duration': f"{stats['duration_minutes']:.1f} minutes"
    }
#
# def _convert_int64_to_int(self, data):
#         """遍历字典，递归地将 numpy.int64 转换为 Python int"""
#         if isinstance(data, dict):
#             return {key: self._convert_int64_to_int(value) for key, value in data.items()}
#         elif isinstance(data, list):
#             return [self._convert_int64_to_int(item) for item in data]
#         elif isinstance(data, np.int64):
#             return int(data)  # 将 numpy.int64 转换为 Python int
#         else:
#             return data
#

import numpy as np
import json
import pandas as pd
import time
from threading import Thread
from queue import Queue
from pathlib import Path
import time
import pynvml
import psutil
from threading import Thread
from queue import Queue
import numpy as np


class ResourceMonitor:
    def __init__(self, log_dir='results/resource_usage'):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 默认使用第一个 GPU
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

        stats = self._convert_int64_to_int(stats)

        with open(self.log_dir / f'{self.tag}_resource_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)

        return stats

    def _convert_int64_to_int(self, data):
        if isinstance(data, dict):
            return {key: self._convert_int64_to_int(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_int64_to_int(item) for item in data]
        elif isinstance(data, np.int64):
            return int(data)
        else:
            return data

    def _monitor_loop(self):
        while self.monitoring:
            try:
                # 获取 GPU 内存使用信息
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                memory_used = info.used / 1024 ** 2  # MB
                memory_total = info.total / 1024 ** 2  # MB

                # 获取 GPU 利用率
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory

                # 获取 GPU 功率使用
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # W

                # 获取 CPU 和内存信息
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent

                # 将这些信息存储到字典中
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

                # 将数据放入队列
                self.metrics_queue.put(metrics)

                # 每0.1秒收集一次数据
                time.sleep(0.1)

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                break
