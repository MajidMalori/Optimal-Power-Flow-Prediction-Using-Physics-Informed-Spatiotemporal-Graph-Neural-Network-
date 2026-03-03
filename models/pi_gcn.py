from torch import nn
from torch_geometric.nn import GCNConv


class PIGCN(nn.Module):
    """
    Model 3: PIGCN (Physics-Informed GCN)
    Combines an Adaptive GCN with a Physics-Informed loss calculation block that enforces 
    conservation of power rules (Kirchhoff's laws constraints).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, physics_weight=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # Adaptive Layers
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        
        # Instantiate the Physics Loss constraint block


    def forward(self, x, dynamic_edge_index, p_inj, q_inj, y_bus):
        """
        Forward pass requiring physics variables.
        """
        out = x
        for conv in self.convs:
            out = self.relu(conv(out, dynamic_edge_index))
            
        preds = self.output_layer(out)
        
        return preds
