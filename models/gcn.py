import torch
import torch.nn as nn
from .base_model import BaseModel
from .layers import VoltageGraphLayer

class GCN(BaseModel):
    def __init__(self,
                 feature_dim: int = 10,
                 hidden_dim: int = 64,
                 num_gc_layers: int = 3,
                 num_buses: int = 118,
                 dropout: float = 0.1):
        
        output_dim = num_buses * 10  # 10 features per bus
        super().__init__(
            feature_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_gc_layers=num_gc_layers, num_buses=num_buses, dropout=dropout
        )
        
        self.gc_layers = nn.ModuleList([
            VoltageGraphLayer(feature_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_gc_layers)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 10)  # All 10 features: V_mag, V_angle, P_load, Q_load, P_ext, Q_ext, P_conv, Q_conv, P_ren, Q_ren

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        
        for gc_layer in self.gc_layers:
            x = torch.relu(gc_layer(x, adj))
            x = self.dropout_layer(x)
        
        out = self.output_layer(x)
        
        # PHYSICAL CONSTRAINTS: Apply ReLU only to parameters that MUST be non-negative
        # Based on physics: vm_pu, p_load, p_conv, p_ren must be positive
        # Can be negative: va_rad, q_load, p_ext, q_ext, q_conv, q_ren (for reactive power control)
        if out.shape[-1] >= 10:  # Ensure we have 10 features
            out[..., 0] = torch.relu(out[..., 0])  # vm_pu ≥ 0 (voltage magnitude always positive)
            out[..., 2] = torch.relu(out[..., 2])  # p_load ≥ 0 (loads consume power)
            out[..., 6] = torch.relu(out[..., 6])  # p_conv ≥ 0 (generators produce power)
            out[..., 8] = torch.relu(out[..., 8])  # p_ren ≥ 0 (renewables produce power)
            # DO NOT apply ReLU to: va_rad [1], q_load [3], p_ext [4], q_ext [5], q_conv [7], q_ren [9]
            # These can be negative for physical reasons (angles, reactive power, slack balancing)
        
        return out.reshape(batch_size, -1)  # Flatten to [batch_size, num_buses * 10]