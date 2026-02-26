import argparse
from src.utils import evaluate_existing_model

def main():
    parser = argparse.ArgumentParser(description='评估已训练的模型')
    parser.add_argument('--model-path', type=str, default='best_model.pth',
                      help='模型文件路径')
    parser.add_argument('--data-path', type=str, default='data/processed/processed_data.pkl',
                      help='数据文件路径')
    
    args = parser.parse_args()
    
    print(f"开始评估模型: {args.model_path}")
    evaluate_existing_model(args.model_path, args.data_path)

if __name__ == '__main__':
    main()