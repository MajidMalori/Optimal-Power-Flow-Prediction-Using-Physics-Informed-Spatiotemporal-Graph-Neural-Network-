import torch
from torch import nn
from torch_geometric.nn import GCNConv

class PIGCLSTM(nn.Module):
    """
    Model 4: PIGCLSTM (Physics-Informed GCN + LSTM)
    Spatial features extracted by PIGCN (dynamic adj), passed sequentially through an LSTM.
    Inputs are expected in sequence shape: (batch_size, sequence_length, num_nodes, num_features)
    """
    def __init__(self, in_channels, gcn_hidden, lstm_hidden, out_channels, num_gcn_layers=2, physics_weight=0.1):
        super().__init__()
        
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, gcn_hidden))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(gcn_hidden, gcn_hidden))
            
        self.relu = nn.ReLU()
        
        # LSTM processes the spatial node embeddings over time
        # Input to LSTM: [batch * num_nodes, sequence_length, gcn_hidden]
        self.lstm = nn.LSTM(input_size=gcn_hidden, hidden_size=lstm_hidden, batch_first=True)
        
        self.output_layer = nn.Linear(lstm_hidden, out_channels)


    def forward(self, x_seq, dynamic_edge_idx_seq, p_inj_final, q_inj_final, y_bus_final):
        """
        x_seq: [batch, seq_len, nodes, features]
        dynamic_edge_idx_seq: list of edge_indices per sequence step
        """
        batch_size, seq_len, num_nodes, num_features = x_seq.shape
        
        # 1. Process Spatial dimension per timestep
        spatial_embeddings = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :, :].reshape(-1, num_features) # [batch*nodes, features]
            edge_index_t = dynamic_edge_idx_seq[t]
            
            out_t = x_t
            for gcn in self.gcn_layers:
                out_t = self.relu(gcn(out_t, edge_index_t))
            spatial_embeddings.append(out_t) # List of [batch*nodes, gcn_hidden]
            
        # Stack properly for LSTM => [batch*nodes, seq_len, gcn_hidden]
        spatial_seq = torch.stack(spatial_embeddings, dim=1)
        
        # 2. Process Temporal dimension
        lstm_out, _ = self.lstm(spatial_seq)
        
        # Take the last hidden state output for prediction
        last_out = lstm_out[:, -1, :] # [batch*nodes, lstm_hidden]
        
        preds = self.output_layer(last_out) # [batch*nodes, out_channels]
        
        # Reshape back to [batch, nodes, out_channels]
        preds = preds.reshape(batch_size, num_nodes, -1)
        
        return preds
