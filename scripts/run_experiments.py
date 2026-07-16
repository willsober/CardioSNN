# src/run_experiments.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.ablation_study import run_ablation_studies
from src.experiments.cross_dataset import cross_dataset_validation
from src.experiments.interpretability import InterpretabilityAnalysis
from src.utils.evaluation import save_results, load_trained_model
from src.data.dataset import load_test_data

def main():

    try:
       
        print("Running ablation studies...")
        ablation_results = run_ablation_studies()
        save_results('ablation', ablation_results)
        
        print("Running cross-dataset validation...")
        cross_dataset_results = cross_dataset_validation()
        save_results('cross_dataset', cross_dataset_results)
        
        print("Running interpretability analysis...")
        model = load_trained_model()
        if model is not None:  
            test_data = load_test_data()
            interpreter = InterpretabilityAnalysis(model)
            
            importance_results = interpreter.feature_importance_analysis(test_data)
            if importance_results:
                save_results('feature_importance', importance_results)
            
           
            process_results = interpreter.visualize_decision_process(test_data)
            if process_results:
                save_results('decision_process', process_results)
        else:
            print("Skipping interpretability analysis due to model loading failure")
        
        print("All experiments completed successfully!")
        
    except Exception as e:
        print(f"Error during experiments: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()