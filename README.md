# CardioSNN-ECG

A CardioSNN spiking neural network framework for arrhythmia classification.



## Project Overview

This project proposes CardioSNN, a lightweight spiking neural network classification model tailored for wearable ECG sensors. Leveraging the inherent advantages of spiking neural networks and integrating a time-aware attention mechanism, the model achieves outstanding classification performance with low power consumption and short inference latency. Its core features are as follows:

- Integrates Spiking Neural Network to reduce computational complexity
- Spike-based Morphology-aware Self-Attention for Efficient Temporal Dynamic Modeling with Linear Complexity
- Achieves accurate classification of three ECG signal types: Normal, Ventricular Ectopic Beat (VEB), and Supraventricular Ectopic Beat (SVEB)

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- scikit-learn
- wfdb
- tqdm

## Installation

```bash
git clone https://github.com/willsober/CardioSNN.git
cd CardioSNN
pip install -r requirements.txt
```

## Data Preparation

1. Download the ECG dataset：

```bash
python download_data.py
```

2. Preprocess data:
```bash
# For PTB-XL dataset
python src_PTB/data/preprocess_data.py

# For MIT-BIH dataset
python src_MIT/data/preprocess_data.py

#The same operation applies to any other self-provided datasets.
```

## Model Training
```bash
python train.py
```
## Project Structure
```bash
CardioSNN/
├── data/ # Data directory
├── src/ # Source code
│ ├── data/ # Data processing modules
│ ├── models/ # Model definitions
│ └── utils/ # Utility functions
├── results/ # Experimental results
└── best_model.pth # Pre-trained model
```





