import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def load_results(results_dir='results'):

    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('_results.pkl'):
            model_name = filename.replace('_results.pkl', '')
            with open(os.path.join(results_dir, filename), 'rb') as f:
                results[model_name] = pickle.load(f)
    return results

def plot_training_curves(results):

    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    for name, result in results.items():
        plt.plot(result['val_losses'], label=f'{name}_val')
        plt.plot(result['train_losses'], '--', label=f'{name}_train')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
 
    plt.subplot(122)
    for name, result in results.items():
        plt.plot(result['val_accs'], label=f'{name}_val')
        plt.plot(result['train_accs'], '--', label=f'{name}_train')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_curves_comparison.png')
    plt.close()

def create_comparison_table(results):
 
    data = []
    for name, result in results.items():
        data.append({
            'Model': name,
            'Best Val Acc': max(result['val_accs']),
            'Test Acc': result['test_acc'],
            'Best Epoch': result['best_epoch'],
            'Training Time': result['training_time']
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    
    results = load_results()
    
   
    plot_training_curves(results)
    
 
    comparison_table = create_comparison_table(results)
    print("\nModel Performance Comparison:")
    print(comparison_table.to_string(index=False))
    
    comparison_table.to_csv('results/model_comparison.csv', index=False)

if __name__ == '__main__':
    main()