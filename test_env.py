import torch
import numpy as np
import wfdb
import spikingjelly.activation_based.neuron as neuron  # 更新导入路径
import matplotlib.pyplot as plt

def test_environment():
    print("\n=== 环境测试 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"NumPy版本: {np.__version__}")
    
    # 测试SpikingJelly功能
    print("\n测试SpikingJelly LIF神经元...")
    try:
        lif = neuron.LIFNode(tau=2.0)
        print("SpikingJelly LIF神经元创建成功!")
    except Exception as e:
        print(f"SpikingJelly测试失败: {e}")
    
    # 测试matplotlib
    print("\n测试Matplotlib...")
    plt.figure(figsize=(3,3))
    plt.plot([1,2,3], [1,2,3])
    plt.title("Matplotlib测试")
    plt.close()
    print("Matplotlib测试成功!")
    
    print("\n所有包测试完成！")

if __name__ == "__main__":
    test_environment()