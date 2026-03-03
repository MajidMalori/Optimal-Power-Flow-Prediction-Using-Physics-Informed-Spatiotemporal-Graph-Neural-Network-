from torch import nn
from torch_geometric.nn import GCNConv

class StandardGCN(nn.Module):
    """
    Model 1: Standard GCN
    Uses a fixed static adjacency matrix.
    Input shape: (batch_size, num_nodes, num_features)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        # x shape: [batch_size * num_nodes, in_channels] for PyG
        # edge_index is fixed for the whole batch
        
        for conv in self.convs:
            x = self.relu(conv(x, edge_index, edge_weight))
            
        out = self.output_layer(x)
        return out
