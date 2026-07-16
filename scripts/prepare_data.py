import os
import pickle
from collections import Counter
import sys
import pathlib


project_root = str(pathlib.Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data.preprocessor import ECGPreprocessor

def main():
  
    data_dir = os.path.join(project_root, 'data', 'raw')
    preprocessor = ECGPreprocessor(data_dir=data_dir)
    
 
    processed_data = preprocessor.process_data()
    
    if processed_data is not None:
      
        processed_dir = os.path.join(project_root, 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
 
        output_path = os.path.join(processed_dir, 'processed_data.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
            
        print("\nData processed and saved successfully!")
        print(f"Saved to: {output_path}")
        print(f"Processed data shape: {processed_data['X_train'].shape}")
        print("\nClass distribution in training set:")
        print(Counter(processed_data['y_train'].numpy()))

if __name__ == "__main__":
    main()