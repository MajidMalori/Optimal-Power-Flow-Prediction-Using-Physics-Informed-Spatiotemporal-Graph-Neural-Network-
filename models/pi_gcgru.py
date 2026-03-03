import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from .layers import PhysicsInformedLoss

class PIGCGRU(nn.Module):
    """
    Model 5: PIGCGRU (Physics-Informed GCN + GRU)
    Faster temporal processing than LSTM using Gated Recurrent Units.
    """
    def __init__(self, in_channels, gcn_hidden, gru_hidden, out_channels, num_gcn_layers=2, physics_weight=0.1):
        super().__init__()
        
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, gcn_hidden))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(gcn_hidden, gcn_hidden))
            
        self.relu = nn.ReLU()
        
        # GRU processes the spatial node embeddings over time
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)
        
        self.output_layer = nn.Linear(gru_hidden, out_channels)
        self.physics_constraint = PhysicsInformedLoss(weight=physics_weight)

    def forward(self, x_seq, dynamic_edge_idx_seq, p_inj_final, q_inj_final, y_bus_final):
        batch_size, seq_len, num_nodes, num_features = x_seq.shape
        
        # 1. Process Spatial dimension per timestep
        spatial_embeddings = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :, :].reshape(-1, num_features)
            edge_index_t = dynamic_edge_idx_seq[t]
            
            out_t = x_t
            for gcn in self.gcn_layers:
                out_t = self.relu(gcn(out_t, edge_index_t))
            spatial_embeddings.append(out_t)
            
        # Stack properly for GRU => [batch*nodes, seq_len, gcn_hidden]
        spatial_seq = torch.stack(spatial_embeddings, dim=1)
        
        # 2. Process Temporal dimension
        gru_out, hn = self.gru(spatial_seq)
        
        # Take the last hidden state output for prediction
        last_out = gru_out[:, -1, :] 
        
        preds = self.output_layer(last_out) 
        
        # Reshape back tracking batch and nodes
        preds = preds.reshape(batch_size, num_nodes, -1)
        
        pred_v = preds[:, :, 0]
        pred_theta = preds[:, :, 1]
        
        physics_loss = self.physics_constraint(pred_v, pred_theta, p_inj_final, q_inj_final, y_bus_final)
        
        return preds, physics_loss
