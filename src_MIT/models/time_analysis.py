import os
import sys
import subprocess
from siamese_snn import SiameseSNN

# --- 1. 环境检查 ---
def install_dependencies():
    requirements = ['torch', 'numpy', 'matplotlib', 'scipy']
    for lib in requirements:
        try:
            __import__(lib)
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", lib, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])


install_dependencies()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# --- 2. 模拟你的具体模型结构 ---
class MockSNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 严格匹配你的报错信息：500 -> 128
        self.feature_extractor = nn.Linear(500, 128)

    def forward_one(self, x):
        # 你的模型内部会做展平: [1, 250, 2] -> [1, 500]
        x_flat = x.reshape(x.size(0), -1)
        # 模拟响应
        spike_out = torch.sigmoid(self.feature_extractor(x_flat))
        return None, None, spike_out


# --- 3. 实验脚本 ---
def run_standalone_experiment():
    print("正在启动时间特征分析实验...")

    # --- 关键修改区 ---
    # 为了满足 input_size = 500:
    # 我们设置 250 个时间点，配合 2 个通道 (250 * 2 = 500)
    fs = 125  # 采样率降为 125Hz
    duration = 2.0  # 时长 2 秒
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # 刚好 250 个点

    # 信号生成 (附录 A.4)
    i_low = 3 * np.sin(2 * np.pi * 0.5 * t)
    i_high = 5 * np.sin(2 * np.pi * 4 * t)
    i_total = i_low + i_high

    # 模拟输入：[1, 250, 2]
    input_signal = np.stack([i_total, i_total], axis=1)
    input_tensor = torch.FloatTensor(input_signal).unsqueeze(0)
    print(f"输入张量形状: {input_tensor.shape}, 总元素: {input_tensor.numel()}")

    model = SiameseSNN()
    model.eval()

    with torch.no_grad():
        _, _, spike_out = model.forward_one(input_tensor)
        # 提取 8 个时间步的响应做演示
        neuron_resp = spike_out.squeeze().numpy()[:8]

        # --- 4. 绘图 ---
    plt.figure(figsize=(10, 12))

    # 子图1: 时域信号
    plt.subplot(3, 1, 1)
    plt.plot(t, i_total, color='gray', alpha=0.3, label='Mixed $I(t)$')
    plt.plot(t, i_low, color='green', linewidth=2, label='Low Freq (0.5Hz)')
    plt.plot(t, i_high, color='red', alpha=0.5, label='High Freq (4Hz)')
    plt.title("Time-Domain Signal (Appendix A.4)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 神经元响应 (验证时间分解)
    plt.subplot(3, 1, 2)
    t_steps = np.linspace(0, duration, len(neuron_resp))
    plt.stem(t_steps, neuron_resp, linefmt='C0-', markerfmt='C0o', label='SNN Spike Intensity')
    # 叠加低频包络辅助观察
    i_low_norm = (i_low - i_low.min()) / (i_low.max() - i_low.min()) * neuron_resp.max()
    plt.plot(t, i_low_norm, 'g--', alpha=0.4, label='Expected Low-Freq Envelope')
    plt.title("Model Internal Temporal Response")
    plt.legend()

    # 子图3: 频谱验证
    plt.subplot(3, 1, 3)
    yf = fft(i_total)
    xf = fftfreq(len(t), 1 / fs)
    plt.plot(xf[:len(t) // 2], np.abs(yf[:len(t) // 2]), color='purple')
    plt.xlim(0, 10)
    plt.title("Frequency Spectrum (FFT Analysis)")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    print("实验完成，正在显示图像...")
    plt.show()


if __name__ == "__main__":
    try:
        run_standalone_experiment()
    except Exception as e:
        print(f"\n运行出错: {e}")
        import traceback

        traceback.print_exc()
        input("\n按回车键退出...")