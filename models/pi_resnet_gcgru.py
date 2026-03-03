import torch
import torch.nn as nn
from .layers import ResidualGCNBlock, PhysicsInformedLoss

class PIResnetGCGRU(nn.Module):
    """
    Model 7: PIResnetGCGRU (Physics-Informed ResNet GCN + GRU)
    Combines the deep Residual GCN layers with the faster GRU temporal layer.
    """
    def __init__(self, in_channels, gcn_hidden, gru_hidden, out_channels, num_res_blocks=3, physics_weight=0.1):
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(ResidualGCNBlock(in_channels, gcn_hidden))
        for _ in range(num_res_blocks - 1):
            self.res_blocks.append(ResidualGCNBlock(gcn_hidden, gcn_hidden))
            
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)
        self.output_layer = nn.Linear(gru_hidden, out_channels)
        self.physics_constraint = PhysicsInformedLoss(weight=physics_weight)

    def forward(self, x_seq, dynamic_edge_idx_seq, p_inj_final, q_inj_final, y_bus_final):
        batch_size, seq_len, num_nodes, num_features = x_seq.shape
        
        spatial_embeddings = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :, :].reshape(-1, num_features)
            edge_index_t = dynamic_edge_idx_seq[t]
            
            out_t = x_t
            for res_block in self.res_blocks:
                out_t = res_block(out_t, edge_index_t)
                
            spatial_embeddings.append(out_t)
            
        spatial_seq = torch.stack(spatial_embeddings, dim=1)
        gru_out, hn = self.gru(spatial_seq)
        
        last_out = gru_out[:, -1, :] 
        preds = self.output_layer(last_out) 
        
        preds = preds.reshape(batch_size, num_nodes, -1)
        
        pred_v = preds[:, :, 0]
        pred_theta = preds[:, :, 1]
        
        physics_loss = self.physics_constraint(pred_v, pred_theta, p_inj_final, q_inj_final, y_bus_final)
        
        return preds, physics_loss
