# Model Evaluation Report

## Noise Robustness Analysis

### Gaussian Noise
| Noise Level | Accuracy | F1 Score |
|-------------|----------|----------|
| 0.05 | 0.9973 | 0.9973 |
| 0.10 | 0.9961 | 0.9961 |
| 0.15 | 0.9960 | 0.9960 |
| 0.20 | 0.9951 | 0.9953 |

### Impulse Noise
| Noise Level | Accuracy | F1 Score |
|-------------|----------|----------|
| 0.05 | 0.9858 | 0.9878 |
| 0.10 | 0.9694 | 0.9754 |
| 0.15 | 0.9514 | 0.9617 |
| 0.20 | 0.9392 | 0.9534 |

### Baseline Noise
| Noise Level | Accuracy | F1 Score |
|-------------|----------|----------|
| 0.05 | 0.9967 | 0.9966 |
| 0.10 | 0.9965 | 0.9965 |
| 0.15 | 0.9966 | 0.9965 |
| 0.20 | 0.9969 | 0.9969 |

## Performance Analysis

### Batch Processing Performance
| Batch Size | Throughput (samples/s) | Latency (ms) |
|------------|----------------------|--------------|
| 1 | 317.96 | 3.15 |
| 4 | 1201.50 | 3.33 |
| 8 | 2395.76 | 3.34 |
| 16 | 4918.41 | 3.25 |
| 32 | 9550.21 | 3.35 |