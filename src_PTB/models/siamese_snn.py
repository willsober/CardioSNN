import torch
import torch.nn as nn
from .snn import SNN

class SiameseSNN(nn.Module):
    def __init__(self, temporal_encoding=True, siamese=True, spike_generation=True):
        super(SiameseSNN, self).__init__()
        
        self.temporal_encoding = temporal_encoding
        self.siamese = siamese
        self.spike_generation = spike_generation
        
        if temporal_encoding:
            self.temporal_encoder = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Flatten() 
            )
            
            self.flattened_size = 64 * 62  
        
        input_size = self.flattened_size if temporal_encoding else 500
        self.base_network = SNN(
            input_size=input_size,
            hidden_size=192,
            num_classes=3,
            time_steps=8 if spike_generation else 1
        )
        
        if siamese:
            self.siamese_network = SNN(
                input_size=input_size,
                hidden_size=192,
                num_classes=3,
                time_steps=8 if spike_generation else 1
            )
            
    def preprocess_input(self, x):
        
        if len(x.shape) == 3:
            batch_size, length, features = x.shape
            if self.temporal_encoding:
             
                x = x.permute(0, 2, 1)
            else:
               
                x = x.reshape(batch_size, -1)
            return x
        
        raise ValueError(f"Unexpected input shape: {x.shape}")
            
    def forward(self, x1, x2=None):
  
        x1 = self.preprocess_input(x1)
        if x2 is not None:
            x2 = self.preprocess_input(x2)
        
       
        if self.temporal_encoding:
            x1 = self.temporal_encoder(x1)
            if x2 is not None:
                x2 = self.temporal_encoder(x2)
        else:
           
            batch_size = x1.size(0)
            x1 = x1.reshape(batch_size, -1)
            if x2 is not None:
                x2 = x2.reshape(batch_size, -1)
        
        embedding1, logits1, spike1 = self.base_network(x1)
        
        if self.siamese and x2 is not None:
            embedding2, logits2, spike2 = self.siamese_network(x2)
          
            similarity = torch.cosine_similarity(embedding1, embedding2, dim=1)
            return logits1, logits2, similarity
        
        return logits1
    
    def get_embedding(self, x):
  
        x = self.preprocess_input(x)
        if self.temporal_encoding:
            x = self.temporal_encoder(x)
        return self.base_network.get_embedding(x) 