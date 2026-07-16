import torch
import pickle
from pathlib import Path
import os
from src.utils.robustness_test import RobustnessEvaluator
from src.utils.visualization import generate_evaluation_report
from src.data.dataset import create_dataloaders
from src.models.snn import SNN
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_curve, auc

def load_trained_model(model_path='best_model.pth'):
   
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        
        from src.models.siamese_snn import SiameseSNN
        model = SiameseSNN(
            temporal_encoding=True,
            siamese=True,
            spike_generation=True
        )
        
       
        if isinstance(checkpoint, dict):
            new_state_dict = {}
            for k, v in checkpoint.items():
                
                if k.startswith('base_network.') or k.startswith('siamese_network.'):
                    new_state_dict[k] = v
                else:
                    
                    if not k.startswith('temporal_encoder.'):
                        new_state_dict[f'base_network.{k}'] = v
                        if model.siamese:
                            new_state_dict[f'siamese_network.{k}'] = v
                    else:
                      
                        if 'encoder' not in k:
                            parts = k.split('.')
                            new_k = f'temporal_encoder.encoder.{parts[-2]}.{parts[-1]}'
                            new_state_dict[new_k] = v
                            if model.siamese:
                                new_state_dict[f'siamese_network.{new_k}'] = v
        
            model.load_state_dict(new_state_dict, strict=False)
            print("Model state dict loaded with key remapping")
        
        model = model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def evaluate_existing_model(model_path='best_model.pth', 
                          data_path='data/processed/processed_data.pkl'):
  
   
    if not Path(model_path).exists():
        raise FileNotFoundError(f": {model_path}")
    if not Path(data_path).exists():
        raise FileNotFoundError(f": {data_path}")
    
    print("...")
    
   
    model = load_trained_model(model_path)
    if model is None:
        return
    
 
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    
 
    _, _, test_loader = create_dataloaders(processed_data, batch_size=32)
    
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    

    results = evaluate_model(model, test_loader, device)
    
 
    save_results('evaluation', results)
    
    return results

def evaluate_model(model, test_loader, device=None):
  
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            x1 = batch['x1'].to(device)
            y1 = batch['y1'].to(device)
            
         
            outputs = model(x1)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            
            all_probs.append(probs.cpu())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y1.cpu().numpy())
    
   
    all_probs = torch.cat(all_probs, dim=0).numpy()
    

    results = {
        'y_true': np.array(all_labels),
        'y_pred': np.array(all_preds),
        'y_pred_proba': all_probs,
        'accuracy': accuracy_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, 
        average='weighted',
        zero_division=0
    )
    
    results.update({
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    return results

def save_results(name, results, base_path='results'):
   
  
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = Path(base_path) / f'{name}_{timestamp}'
    result_dir.mkdir(parents=True, exist_ok=True)
    

    result_path = result_dir / 'raw_results.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    
    
    metrics_path = result_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                f.write(f'{metric}: {value:.4f}\n')
    

    if 'confusion_matrix' in results:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'AF', 'Other'],
            yticklabels=['Normal', 'AF', 'Other']
        )
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(result_dir / 'confusion_matrix.png')
        plt.close()
    

    if 'y_true' in results and 'y_pred_proba' in results:
        plt.figure(figsize=(10, 8))
        for i in range(3):  
            fpr, tpr, _ = roc_curve(
                (results['y_true'] == i).astype(int),
                results['y_pred_proba'][:, i]
            )
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                label=f'Class {i} (AUC = {roc_auc:.2f})'
            )
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{name} ROC Curves')
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / 'roc_curves.png')
        plt.close()
    
    return result_dir
