import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DynamicGCN(nn.Module):
    """
    Model 2: Dynamic GCN
    Uses the real-time post-contingency adjacency matrix for message passing.
    When a line trips, message flow between disconnected buses stops immediately.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, dynamic_edge_index, dynamic_edge_weight=None):
        """
        Forward pass with real-time topology.
        dynamic_edge_index changes per timestep based on contingency state.
        """
        for conv in self.convs:
            x = self.relu(conv(x, dynamic_edge_index, dynamic_edge_weight))
            
        return self.output_layer(x)
