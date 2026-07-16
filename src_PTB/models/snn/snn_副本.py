import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, surrogate, layer
from .snn_layers import AdaptiveLIFLayer, TemporalEncoder  

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        return x + self.block(x)

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=3, time_steps=8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.time_steps = time_steps
        
        self.temporal_encoder = TemporalEncoder(input_size)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            ResidualBlock(hidden_size),
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.GELU()
        )
        
        hidden_dims = [hidden_size//2, hidden_size//4, hidden_size//8]
        self.snn_layers = nn.ModuleList([
            nn.Sequential(
                AdaptiveLIFLayer(hidden_dims[i]),
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1])
            ) for i in range(len(hidden_dims)-1)
        ])
        
        attention_dim = hidden_dims[-1]  # hidden_size//8
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=4,
            batch_first=True
        )
        
        
        self.metric_layer = nn.Sequential(
            nn.Linear(attention_dim, attention_dim//2),
            nn.LayerNorm(attention_dim//2),
            nn.GELU(),
            nn.Linear(attention_dim//2, attention_dim//4)
        )
        
      
        self.classifier = nn.Linear(attention_dim//4, num_classes)
        
    def forward_one(self, x):
        
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        temporal_x = self.temporal_encoder(x, self.time_steps)  # [batch, time_steps, input_size]
        
        features = self.feature_extractor(temporal_x)  # [batch, time_steps, hidden_size//2]
        
        spikes = features
        for snn_layer in self.snn_layers:
            spikes = snn_layer(spikes)
        
        spike_out = spikes.mean(dim=1)  # [batch, hidden_size//8]
        
        attn_input = spike_out.unsqueeze(1)  # [batch, 1, hidden_size//8]
        attn_output, _ = self.attention(
            attn_input,
            attn_input,
            attn_input
        )  # [batch, 1, hidden_size//8]
        features = attn_output.squeeze(1)  # [batch, hidden_size//8]
        
        embeddings = self.metric_layer(features)  # [batch, hidden_size//32]
        logits = self.classifier(embeddings)  # [batch, num_classes]
        
        return embeddings, logits, spike_out
        
    def forward(self, x1, x2=None):
        if x2 is not None:
            embedding1, logits1, spike1 = self.forward_one(x1)
            embedding2, logits2, spike2 = self.forward_one(x2)
            return embedding1, embedding2, logits1, logits2, spike1, spike2
        
        embedding1, logits1, spike1 = self.forward_one(x1)
        return embedding1, logits1, spike1

class CombinedLoss(nn.Module):
    def __init__(self, margin=2.0, alpha=0.65, beta=0.1, l2_reg=0.01):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.l2_reg = l2_reg
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, embeddings1, embeddings2, logits1, logits2, 
                labels1, labels2, spike1, spike2, model=None):
       
        ce_loss = self.ce_loss(logits1, labels1) + self.ce_loss(logits2, labels2)
        
        
        distance = F.pairwise_distance(embeddings1, embeddings2)
        same_class = (labels1 == labels2).float()
        contrastive_loss = same_class * distance.pow(2) + \
                          (1 - same_class) * F.relu(self.margin - distance).pow(2)
        
        
        sparsity_loss = (spike1.mean() + spike2.mean()) / 2
        
       
        l2_loss = 0
        if model is not None:
            for param in model.parameters():
                l2_loss += torch.norm(param, 2)
        
        return self.alpha * ce_loss + \
               (1 - self.alpha) * contrastive_loss.mean() + \
               self.beta * sparsity_loss + \
               self.l2_reg * l2_loss