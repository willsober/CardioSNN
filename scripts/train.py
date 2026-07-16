# scripts/train.py
import torch
import pickle
from src.models.snn import SNN, CombinedLoss
from src.data.dataset import create_dataloaders
from src.utils.training import train_model

def main():
  
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    with open('../data/processed/processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    model, history = train_model(
        processed_data,
        num_epochs=100,
        batch_size=128,
        lr=0.001
    )
    
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

if __name__ == '__main__':
    main()