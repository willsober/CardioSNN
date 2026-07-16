# src/experiments/interpretability.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, LayerGradCam
import seaborn as sns

class InterpretabilityAnalysis:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device if model is not None else None
        if model is not None:
            self.ig = IntegratedGradients(model)
            self.grad_cam = LayerGradCam(model, model.base_network.feature_extractor[-1])
    
    def calculate_feature_importance(self, model, input_data):
        """Calculate feature importance"""
        if model is None:
            return None
            
        importances = []
        for batch in input_data:
            x = batch['x1'].to(self.device).requires_grad_(True)
            y = batch['y1'].to(self.device)
            
            try:
                attributions = self.ig.attribute(x, target=y)
                importance = torch.mean(torch.abs(attributions), dim=0)
                importances.append(importance.detach().cpu())
            except Exception as e:
                print(f"Error during attribution: {str(e)}")
                continue
        
        if importances:
            return torch.mean(torch.stack(importances), dim=0)
        return None
    
    def visualize_importance(self, importance_scores):
        """Visualize feature importance"""
        if importance_scores is None:
            return None
            
        if isinstance(importance_scores, torch.Tensor):
            importance_scores = importance_scores.detach().cpu().numpy()
            
        plt.figure(figsize=(10, 6))
        sns.heatmap(importance_scores, cmap='viridis')
        plt.title('Feature Importance Heatmap')
        plt.xlabel('Time Steps')
        plt.ylabel('Features')
        return plt.gcf()
    
    def feature_importance_analysis(self, input_data):
        """Feature importance analysis"""
        if self.model is None:
            return {}
            
        importance_scores = self.calculate_feature_importance(
            self.model, input_data
        )
        visualization = self.visualize_importance(importance_scores)
        
        return {
            'importance_scores': importance_scores.detach().cpu() if importance_scores is not None else None,
            'visualization': visualization
        }
    
    def get_layer_activations(self, model, input_data):
        """Get layer activations"""
        if model is None:
            return None
            
        activations = []
        def hook(module, input, output):
            # Ensure output is 2D
            if len(output.shape) == 3:
                # [batch, channels, time] -> [batch, channels*time]
                output = output.reshape(output.size(0), -1)
            activations.append(output.detach().cpu())
            
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                handles.append(module.register_forward_hook(hook))
                
        with torch.no_grad():
            for batch in input_data:
                x = batch['x1'].to(self.device)
                model(x)
                
        for handle in handles:
            handle.remove()
            
        return activations
    
    def generate_attention_maps(self, activations):
        """Generate attention maps"""
        if not activations:
            return None
            
        attention_maps = []
        for activation in activations:
            # Ensure output is 2D
            if len(activation.shape) == 3:
                # [batch, channels, time] -> [batch, channels*time]
                attention = activation.reshape(activation.size(0), -1)
            else:
                attention = activation
            attention_maps.append(attention)
        return attention_maps
    
    def visualize_spike_patterns(self, activations):
        """Visualize spike patterns"""
        if activations is None:
            return None
            
        patterns = []
        for activation in activations:
            if len(activation.shape) == 3:  # [batch, time, features]
                spike_pattern = torch.mean(activation, dim=0)
                patterns.append(spike_pattern)
        return patterns
    
    def visualize_decision_process(self, input_data):
        """Visualize decision process"""
        if self.model is None:
            return {}
        
        # Get a batch of data for analysis
        batch = next(iter(input_data))
        x = batch['x1'].to(self.device)
        
        # 1. Feature importance
        importance = self.calculate_feature_importance(self.model, [batch])
        
        # 2. Neuron activation patterns
        activations = self.get_layer_activations(self.model, [batch])
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Input signal
        if len(x.shape) == 3:  # [batch, time, features]
            signal = x[0].cpu().numpy()
            axes[0,0].plot(signal)
            axes[0,0].set_title('Input Signal')
        
        # 2. Feature importance
        if importance is not None:
            importance_2d = importance.view(-1, importance.size(-1)).cpu()
            sns.heatmap(importance_2d, ax=axes[0,1], cmap='viridis')
            axes[0,1].set_title('Feature Importance')
        
        # 3. Activation patterns
        if activations:
            act = activations[0].mean(dim=0).cpu()  # Average across batches
            if len(act.shape) > 2:
                act = act.mean(dim=0)  # If necessary, further reduce dimensions
            sns.heatmap(act.unsqueeze(0), ax=axes[1,0], cmap='viridis')
            axes[1,0].set_title('Average Activation')
        
        # 4. Prediction confidence
        with torch.no_grad():
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            
        confidence = probs[0].cpu().numpy()
        axes[1,1].bar(['Normal', 'AF', 'Other'], confidence)
        axes[1,1].set_title('Prediction Confidence')
        
        plt.tight_layout()
        
        return {
            'importance': importance,
            'activations': activations,
            'visualization': fig
        }
    
    def analyze_errors(self, test_data, predictions):
        """Error case analysis"""
        if self.model is None:
            return {}
            
        results = {
            'error_cases': [],
            'error_patterns': {},
            'confusion_matrix': None
        }
        
        return results