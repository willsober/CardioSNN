# src/experiments/cross_dataset.py
import torch
from pathlib import Path
from src.utils.evaluation import evaluate_model, load_trained_model
from src.data.dataset import ECGDataset
from torch.utils.data import DataLoader
import wfdb
import numpy as np
import pickle

def load_mitbih_data(data_dir='data/raw'):
    """Load MIT-BIH dataset"""
    print("\nLoading MIT-BIH dataset...")
    try:
        # Use processed data
        with open('data/processed/processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except:
        print("Could not load processed MIT-BIH data")
        return None

def create_subset_validation(data, subset_size=0.3, seed=42):
    """Create data subset for validation"""
    np.random.seed(seed)
    total_samples = len(data['X_test'])
    subset_indices = np.random.choice(
        total_samples, 
        size=int(total_samples * subset_size), 
        replace=False
    )
    
    return {
        'X_test': data['X_test'][subset_indices],
        'y_test': data['y_test'][subset_indices]
    }

def cross_dataset_validation(model_path='best_model.pth'):
    """Cross-dataset validation"""
    print("\nStarting cross-dataset validation...")
    results = {}
    
    # Load model
    model = load_trained_model(model_path)
    if model is None:
        print("Failed to load model")
        return {}
        
    device = next(model.parameters()).device
    print(f"Model loaded successfully. Using device: {device}")
    
    # Load MIT-BIH dataset
    mitbih_data = load_mitbih_data()
    if mitbih_data is not None:
        # Create three different subsets for validation
        for i in range(3):
            subset_data = create_subset_validation(
                mitbih_data, 
                subset_size=0.3, 
                seed=42+i
            )
            
            test_dataset = ECGDataset(
                subset_data['X_test'],
                subset_data['y_test'],
                augment=False
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=0
            )
            
            # Evaluate model
            print(f"\nEvaluating on MIT-BIH subset {i+1}...")
            metrics = evaluate_model(model, test_loader)
            results[f'mitbih_subset_{i+1}'] = metrics
            
            print(f"Subset {i+1} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
    
    print("\nCross-dataset validation completed!")
    return results