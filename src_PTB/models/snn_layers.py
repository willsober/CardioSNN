import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, surrogate, layer

class AdaptiveLIFLayer(nn.Module):
    def __init__(self, size, tau=2.0, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.lif = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate.ATan()
        )
        self.threshold = nn.Parameter(torch.ones(1, size) * v_threshold)
        
    def forward(self, x):
        batch_size = x.size(0)
        time_steps = x.size(1)
        spikes = []
        self.lif.reset()
        
        threshold = self.threshold.expand(batch_size, -1)
        
        for t in range(time_steps):
            current_x = x[:, t]  # [batch, size]
            spike = self.lif(current_x)
            spikes.append(spike)
        
        return torch.stack(spikes, dim=1)  # [batch, time_steps, size]

class TemporalEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, time_steps=8):

        if len(x.shape) == 3:
            batch_size, seq_len, channels = x.shape
            x = x.view(batch_size, seq_len * channels)
            
        encoded = self.encoder(x)  # [batch, input_size]
        encoded = self.dropout(encoded)
        temporal = encoded.unsqueeze(1).repeat(1, time_steps, 1)  # [batch, time_steps, input_size]
        return temporal

class SNNLayer(nn.Module):
    def __init__(self, input_size, output_size, time_steps=8):
        super(SNNLayer, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.time_steps = time_steps
        
        self.threshold = 1.0
       
        self.decay = 0.2
        
       
        self.weight = nn.Parameter(torch.Tensor(output_size, input_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        
       
        nn.init.kaiming_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)
        
    def forward(self, x):
        batch_size = x.size(0)
        
  
        membrane_potential = torch.zeros(batch_size, self.output_size, device=x.device)
        spikes = []
 
        for t in range(self.time_steps):
         
            current = F.linear(x, self.weight, self.bias)
            
            membrane_potential = membrane_potential * self.decay + current
            
            spike = (membrane_potential >= self.threshold).float()
            spikes.append(spike)
            
            membrane_potential = membrane_potential * (1 - spike)
        
        spikes = torch.stack(spikes, dim=1)
        
        return spikes.mean(dim=1)
    
    def reset_membrane_potential(self):

        self.membrane_potential = None
        
    def extra_repr(self):
       
        return f'input_size={self.input_size}, output_size={self.output_size}, time_steps={self.time_steps}'