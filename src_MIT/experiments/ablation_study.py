# src/experiments/ablation_study.py
import torch
from src.models.siamese_snn import SiameseSNN
from src.utils.evaluation import evaluate_model, load_trained_model
from src.data.dataset import create_dataloaders
import pickle
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_ablation_studies(data_path='data/processed/processed_data.pkl'):
 
    print("\nStarting ablation studies...")
    results = {}
    
    pretrained = load_trained_model()
    if pretrained is not None:
        pretrained_state = pretrained.state_dict()
    
    # Create data loader
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    _, _, test_loader = create_dataloaders(processed_data)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Test different model variants
    variants = [
        ('full_model', dict(temporal_encoding=True, siamese=True, spike_generation=True)),
        ('no_temporal', dict(temporal_encoding=False, siamese=True, spike_generation=True)),
        ('no_siamese', dict(temporal_encoding=True, siamese=False, spike_generation=True)),
        ('no_spike', dict(temporal_encoding=True, siamese=True, spike_generation=False))
    ]
    
    for name, config in variants:
        print(f"\nTesting {name}...")
        model = SiameseSNN(**config).to(device)
        
        # Load pretrained weights if available
        if pretrained is not None:
            try:
                model.load_state_dict(pretrained_state, strict=False)
            except Exception as e:
                print(f"Warning: Could not load pretrained weights for {name}: {e}")
        
        results[name] = evaluate_model(model, test_loader)
    
    return results

def analyze_ablation_results(results):
    """Analyze ablation study results"""
    # Create comparison table
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    comparison = pd.DataFrame(columns=metrics)
    
    for model_name, result in results.items():
        comparison.loc[model_name] = [
            result.get(metric, 0) for metric in metrics
        ]
    
    # Calculate relative performance drop
    baseline = comparison.loc['full_model']
    relative_drop = (baseline - comparison) / baseline * 100
    
    # Plot performance comparison
    plt.figure(figsize=(12, 6))
    comparison.plot(kind='bar')
    plt.title('Performance Comparison Across Models')
    plt.xlabel('Model Variant')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot performance drop heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        relative_drop.iloc[1:],  # Exclude baseline model
        annot=True,
        fmt='.1f',
        cmap='RdYlBu_r'
    )
    plt.title('Performance Drop Relative to Full Model (%)')
    plt.tight_layout()
    
    return {
        'comparison': comparison,
        'relative_drop': relative_drop,
        'figures': plt.gcf()
    }