"""
Forensic-instrumented wrapper for GCN model.
This version logs everything happening in the forward pass.
"""

import torch
import torch.nn as nn
from models.gcn import GCN as OriginalGCN

class ForensicGCN(OriginalGCN):
    """
    GCN model with forensic logging instrumentation.
    Inherits from original GCN but adds logging at every step.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forensic_logger = None
        self.forward_count = 0
    
    def set_logger(self, logger):
        """Attach a forensic logger."""
        self.forensic_logger = logger
    
    def forward(self, x, adj, bus_types=None):
        """
        Forward pass with forensic logging.
        
        Args:
            x: Input features [batch, buses, input_dim]
            adj: Adjacency matrix [buses, buses] or [batch, buses, buses]
            bus_types: Optional bus type codes [batch, buses]
        
        Returns:
            outputs: Model predictions [batch, buses, output_dim]
        """
        self.forward_count += 1
        
        # Log input
        if self.forensic_logger and self.forward_count % 10 == 1:  # Log every 10th forward pass
            self.forensic_logger.log_model_forward(
                "GCN_INPUT",
                {
                    'features': x,
                    'adjacency': adj,
                    'bus_types': bus_types
                },
                None  # Output not computed yet
            )
            
            # Log layer-by-layer
            self.forensic_logger.logger.debug(f"\n  GCN FORWARD PASS #{self.forward_count}:")
            self.forensic_logger.log_tensor_stats("Input features", x, indent=2)
        
        # Call original forward pass from parent class
        outputs = super().forward(x, adj, bus_types)
        
        # Log output
        if self.forensic_logger and self.forward_count % 10 == 1:
            self.forensic_logger.log_tensor_stats("Final output", outputs, indent=2)
            
            # Check for model collapse
            if outputs.std().item() < 1e-6:
                self.forensic_logger.log_diagnosis(
                    f"MODEL COLLAPSE in forward pass #{self.forward_count}: "
                    f"Output std = {outputs.std().item():.2e}"
                )
        
        return outputs
