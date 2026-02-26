# 导入必要的库
import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate
# spikingjelly 仅做占位，其编码器需单独实现
import spikingjelly as sj


# 定义三种脉冲编码器
class RepeatEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.out_size = output_size
        self.lif = snn.Leaky(
            beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C  (L: 序列长度, C: 特征维度)
        # 修正repeat维度错误：原代码repeat维度顺序错误，导致形状异常
        # 正确逻辑：将输入在时间步维度重复output_size次
        repeat_dims = [1] * len(inputs.size())  # 初始化为全1 (batch, L, C)
        repeat_dims[1] = self.out_size  # 只在L维度重复: (batch, out_size*L, C)
        inputs = inputs.repeat(repeat_dims)

        # 调整维度以适配LIF: (batch, C, out_size*L) → 符合LIF的输入格式
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, C, out_size*L)
        spks = self.lif(inputs)
        return spks


class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),  # 保持序列长度不变
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = snn.Leaky(
            beta=0.99,
            spike_grad=surrogate.atan(alpha=2.0),
            init_hidden=True,
            output=False,
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, C, L) → 适配Conv2d输入
        enc = self.encoder(inputs)  # (batch, output_size, C, L)
        spks = self.lif(enc)
        return spks


class DeltaEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.enc = nn.Linear(1, output_size)
        self.lif = snn.Leaky(
            beta=0.99, spike_grad=surrogate.atan(), init_hidden=True, output=False
        )

    def forward(self, inputs: torch.Tensor):
        # inputs: batch, L, C
        # 计算差分：当前步 - 前一步（捕捉时序变化）
        delta = torch.zeros_like(inputs)
        delta[:, 1:] = inputs[:, 1:, :] - inputs[:, :-1, :]

        # 维度调整 + 归一化
        delta = delta.unsqueeze(1).permute(0, 1, 3, 2)  # (batch, 1, C, L)
        delta = self.norm(delta)

        # 线性映射到目标维度
        delta = delta.permute(0, 2, 3, 1)  # (batch, C, L, 1)
        enc = self.enc(delta)  # (batch, C, L, output_size)
        enc = enc.permute(0, 3, 1, 2)  # (batch, output_size, C, L)

        spks = self.lif(enc)
        return spks


# 构建编码器字典（修正spikingjelly部分的无效引用）
# 注：spikingjelly没有现成的Repeat/Conv/DeltaEncoder，需自行实现后替换占位符
SpikeEncoder = {
    "snntorch": {
        "repeat": RepeatEncoder,
        "conv": ConvEncoder,
        "delta": DeltaEncoder,
    },
    "spikingjelly": {
        "repeat": None,  # 需自行基于spikingjelly实现
        "conv": None,  # 需自行基于spikingjelly实现
        "delta": None,  # 需自行基于spikingjelly实现
    },
}

# # 测试代码（可运行，验证编码器功能）
# if __name__ == "__main__":
#     # 模拟输入：batch=2, 序列长度L=10, 特征维度C=5
#     x = torch.randn(2, 10, 5)
#
#     # 测试RepeatEncoder
#     repeat_enc = RepeatEncoder(output_size=8)
#     repeat_spks = repeat_enc(x)
#     print("RepeatEncoder输出形状:", repeat_spks.shape)  # 预期: (2, 8, 5, 80)
#
#     # 测试ConvEncoder
#     conv_enc = ConvEncoder(output_size=8, kernel_size=3)
#     conv_spks = conv_enc(x)
#     print("ConvEncoder输出形状:", conv_spks.shape)  # 预期: (2, 8, 5, 10)
#
#     # 测试DeltaEncoder
#     delta_enc = DeltaEncoder(output_size=8)
#     delta_spks = delta_enc(x)
#     print("DeltaEncoder输出形状:", delta_spks.shape)  # 预期: (2, 8, 5, 10)