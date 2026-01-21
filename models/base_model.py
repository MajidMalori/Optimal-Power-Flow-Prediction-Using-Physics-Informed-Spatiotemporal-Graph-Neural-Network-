import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int, num_gc_layers: int, num_buses: int, rnn_type: str = None, rnn_layers: int = 0, physics_informed: bool = False, dropout: float = 0.1, **kwargs):
        # Only call super() if not in multiple inheritance scenario
        # In multiple inheritance, SpatioTemporalBase will handle nn.Module initialization
        if not hasattr(self, '_skip_super_init'):
            super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gc_layers = num_gc_layers
        self.num_buses = num_buses
        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers
        self.physics_informed = physics_informed
        self.dropout = dropout
    @abstractmethod
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        pass
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)