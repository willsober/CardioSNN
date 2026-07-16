import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, surrogate, layer

class AdaptiveThresholdLIF(neuron.LIFNode):
    def __init__(self, tau=2.0, v_threshold=1.0, v_reset=0.0):
        super().__init__(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        self.threshold = nn.Parameter(torch.tensor(v_threshold))
        
    def forward(self, x):
        self.v = self.v * self.tau + x
        spike = (self.v >= self.threshold).float()
        self.v = self.v * (1 - spike) + self.v_reset * spike
        return spike

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels//3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels//3, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels//3, kernel_size=7, padding=3)
        
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        return torch.cat([out1, out2, out3], dim=1)

class LightweightBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwconv = nn.Conv1d(in_channels, in_channels, kernel_size=3, 
                               padding=1, groups=in_channels)
        self.pwconv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        out = self.dwconv(x)
        out = self.pwconv(out)
        return out