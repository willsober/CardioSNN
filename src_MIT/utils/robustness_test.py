import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

class RobustnessEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.noise_types = {
            'gaussian': self.add_gaussian_noise,
            'impulse': self.add_impulse_noise,
            'baseline': self.add_baseline_wander
        }
    
    def add_gaussian_noise(self, signal, level):
        
        noise = torch.randn_like(signal) * level
        return signal + noise
    
    def add_impulse_noise(self, signal, level):
      
        mask = torch.rand_like(signal) < level
        noise = torch.randn_like(signal) * 2
        return signal + mask * noise
    
    def add_baseline_wander(self, signal, level):
      
        t = torch.linspace(0, 1, signal.shape[-1])
        wander = torch.sin(2 * np.pi * t) * level
        return signal + wander.to(signal.device)
    
    @torch.no_grad()
    def evaluate_noise_robustness(self, test_loader, noise_levels=[0.1, 0.2, 0.3]):
      
        results = {}
        
        for noise_type in self.noise_types.keys():
            noise_results = {}
            for level in noise_levels:
                all_preds = []
                all_labels = []
                
                for batch in tqdm(test_loader, desc=f'{noise_type} noise {level}'):
                    x1, x2 = batch['x1'], batch['x2']
                    y1, y2 = batch['y1'], batch['y2']
                    
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    
                  
                    noisy_x1 = self.noise_types[noise_type](x1, level)
                    noisy_x2 = self.noise_types[noise_type](x2, level)
                    
                  
                    _, logits1, _ = self.model(noisy_x1)
                    _, logits2, _ = self.model(noisy_x2)
                    
                    preds1 = logits1.argmax(dim=1).cpu()
                    preds2 = logits2.argmax(dim=1).cpu()
                    
                    all_preds.extend(preds1.numpy())
                    all_preds.extend(preds2.numpy())
                    all_labels.extend(y1.numpy())
                    all_labels.extend(y2.numpy())
                
                
                acc = accuracy_score(all_labels, all_preds)
                prec, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, all_preds, average='weighted'
                )
                
                noise_results[level] = {
                    'accuracy': acc,
                    'precision': prec,
                    'recall': recall,
                    'f1': f1
                }
            
            results[noise_type] = noise_results
        
        return results

    def evaluate_cross_dataset(self, other_dataset_loader):
      
        pass  

    def evaluate_realtime_performance(self, input_size=(1, 500), batch_sizes=[1, 4, 8, 16, 32]):
    
        results = {}
        for batch_size in batch_sizes:
       
            dummy_input = torch.randn(batch_size, *input_size).to(self.device)
        
            times = []
            for _ in range(100): 
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model(dummy_input)
                times.append((time.time() - start_time) * 1000)  
            
    
            mean_time = np.mean(times)
            throughput = (batch_size * 1000) / mean_time 
            
            results[batch_size] = {
                'mean_time': mean_time,
                'throughput': throughput
            }
        
        return results