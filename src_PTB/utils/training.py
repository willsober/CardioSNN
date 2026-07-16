import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from datetime import datetime
import os

from src.models.snn import SNN, CombinedLoss
from src.data.dataset import create_dataloaders

class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.0008):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop = False
        
    def step(self, val_loss, val_acc):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_acc = val_acc
        elif val_loss > self.best_loss - self.min_delta and val_acc <= self.best_acc:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = min(val_loss, self.best_loss)
            self.best_acc = max(val_acc, self.best_acc)
            self.counter = 0
        return self.early_stop

def train_epoch(model, criterion, optimizer, train_loader, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
        y1, y2 = batch['y1'].to(device), batch['y2'].to(device)
 
        embedding1, embedding2, logits1, logits2, spike1, spike2 = model(x1, x2)
        
  
        loss = criterion(embedding1, embedding2, logits1, logits2, 
                        y1, y2, spike1, spike2, model=model)
        
  
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
      
        pred1 = logits1.argmax(dim=1)
        pred2 = logits2.argmax(dim=1)
        correct += (pred1 == y1).sum().item() + (pred2 == y2).sum().item()
        total += len(y1) * 2
        
       
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f'{total_loss/(progress_bar.n+1):.4f}',
            'acc': f'{100.0*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total, None, None

def validate_epoch(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
            y1, y2 = batch['y1'].to(device), batch['y2'].to(device)
            
            embedding1, embedding2, logits1, logits2, spike1, spike2 = model(x1, x2)
            
            loss = criterion(embedding1, embedding2, logits1, logits2, y1, y2, spike1, spike2)
            
            pred1 = logits1.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)
            correct += (pred1 == y1).sum().item() + (pred2 == y2).sum().item()
            total += len(y1) * 2
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader), correct / total

def test_model(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
            y1, y2 = batch['y1'].to(device), batch['y2'].to(device)
            
            
            embedding1, embedding2, logits1, logits2, spike1, spike2 = model(x1, x2)
            
       
            loss = criterion(embedding1, embedding2, logits1, logits2, y1, y2, spike1, spike2)
            
            
            pred1 = logits1.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)
            correct += (pred1 == y1).sum().item() + (pred2 == y2).sum().item()
            total += len(y1) * 2
            
           
            all_preds.extend(pred1.cpu().numpy())
            all_preds.extend(pred2.cpu().numpy())
            all_labels.extend(y1.cpu().numpy())
            all_labels.extend(y2.cpu().numpy())
            
            total_loss += loss.item()
    
    return total_loss / len(test_loader), correct / total, all_preds, all_labels

def plot_metrics(train_losses, val_losses, train_accs, val_accs, model_name='default'):
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy over epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_metrics.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, model_name='default'):
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/{model_name}_confusion_matrix.png')
    plt.close()

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, save_path='results/training_metrics.png'):
   
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(train_accs, label='Train Acc', color='blue')
    plt.plot(val_accs, label='Val Acc', color='red')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_overfitting_metrics(train_losses, val_losses, train_accs, val_accs):
 
    last_epochs = 10
    recent_train_loss = np.mean(train_losses[-last_epochs:])
    recent_val_loss = np.mean(val_losses[-last_epochs:])
    recent_train_acc = np.mean(train_accs[-last_epochs:])
    recent_val_acc = np.mean(val_accs[-last_epochs:])
    
    loss_gap = recent_train_loss - recent_val_loss
    acc_gap = recent_train_acc - recent_val_acc
    
    val_loss_std = np.std(val_losses[-last_epochs:])
    val_acc_std = np.std(val_accs[-last_epochs:])
    
    return {
        'loss_gap': loss_gap,
        'acc_gap': acc_gap,
        'val_loss_std': val_loss_std,
        'val_acc_std': val_acc_std,
        'train_loss': recent_train_loss,
        'val_loss': recent_val_loss,
        'train_acc': recent_train_acc,
        'val_acc': recent_val_acc
    }

def train_model(processed_data, model_name='default', num_epochs=100, batch_size=128, lr=0.001, time_steps=8):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_grad_norm = 1.0
    print(f"\nUsing device: {device}")
    
    X_train = processed_data['X_train']
    input_size = X_train.shape[1] * X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
    num_classes = len(processed_data['classes'])
    
    print("\nData dimension check:")
    print(f"Feature shape: {tuple(X_train.shape)}")
    print(f"Label shape: {processed_data['y_train'].shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Input size after flattening: {input_size}")
    
    model = SNN(
        input_size=input_size, 
        hidden_size=192, 
        num_classes=num_classes,
        time_steps=time_steps
    ).to(device)
    
    criterion = CombinedLoss(margin=2.0, alpha=0.65, beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    num_training_steps = len(processed_data['X_train']) // batch_size * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=num_training_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_data, batch_size=batch_size
    )
    
    print("\nFirst batch dimensions:")
    first_batch = next(iter(train_loader))
    print(f"x1 shape: {first_batch['x1'].shape}")
    print(f"x2 shape: {first_batch['x2'].shape}")
    print(f"y1 shape: {first_batch['y1'].shape}")
    print(f"y2 shape: {first_batch['y2'].shape}")
    
    best_val_acc = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=12, min_delta=0.0008)
    start_time = datetime.now()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
       
        train_loss, train_acc, _, _ = train_epoch(
            model, criterion, optimizer, train_loader, device, max_grad_norm)
        
     
        val_loss, val_acc = validate_epoch(model, criterion, val_loader, device)
        
     
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
       
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
       
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
        
        
        if early_stopping.step(val_loss, val_acc):
            print("\nEarly stopping triggered!")
            break
    
    
    plot_metrics(train_losses, val_losses, train_accs, val_accs, model_name)
    
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
   
    test_loss, test_acc, test_preds, test_labels = test_model(model, criterion, test_loader, device)
    
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                              target_names=processed_data['classes']))
    
    plot_confusion_matrix(test_labels, test_preds, processed_data['classes'], model_name)
    
    overfitting_metrics = calculate_overfitting_metrics(
        train_losses, val_losses, train_accs, val_accs
    )
    
    print("\nOverfitting Analysis:")
    print(f"Training vs Validation Loss Gap: {overfitting_metrics['loss_gap']:.4f}")
    print(f"Training vs Validation Accuracy Gap: {overfitting_metrics['acc_gap']:.4f}")
    print(f"Validation Loss Stability (std): {overfitting_metrics['val_loss_std']:.4f}")
    print(f"Validation Accuracy Stability (std): {overfitting_metrics['val_acc_std']:.4f}")
    
    overfitting_threshold = 0.1  
    if (overfitting_metrics['acc_gap'] > overfitting_threshold or 
        overfitting_metrics['loss_gap'] > overfitting_threshold):
        print("\nWarning: Model might be overfitting!")
        print("Consider:")
        print("1. Increasing dropout rate")
        print("2. Adding L2 regularization")
        print("3. Reducing model complexity")
        print("4. Using early stopping with stricter criteria")
        print("5. Applying more data augmentation")
    
   
    results = {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'best_epoch': checkpoint['epoch'],
        'training_time': (datetime.now() - start_time).total_seconds() / 60,
        'model_config': {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_classes': model.num_classes,
            'alpha': criterion.alpha,
            'margin': criterion.margin
        },
        'overfitting_metrics': overfitting_metrics
    }
    
    plot_training_metrics(
        train_losses, val_losses, train_accs, val_accs,
        save_path=f'results/{model_name}_training_metrics.png'
    )
    
    with open(f'results/{model_name}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    return model, results