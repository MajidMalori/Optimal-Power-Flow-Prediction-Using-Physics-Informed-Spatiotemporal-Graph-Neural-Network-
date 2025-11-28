from models.gcn import GCN
from models.adaptive_gcn import adaptiveGCN
from models.adaptive_pigcn import AdaptivePIGCN
from models.pigc_rnn import PIGCLSTM, PIGCGRU, ResnetPIGCLSTM, ResnetPIGCGRU

# Expose new names directly
StandardGraphNet = GCN
AdaptiveGraphNet = adaptiveGCN
PhysicsInformedGraphNet = AdaptivePIGCN
PhysicsInformedRecurrentNet = PIGCLSTM # Base class technically, but exposed for reference

# For backward compatibility with scripts that import by string name
__all__ = [
    'GCN', 'adaptiveGCN', 'AdaptivePIGCN',
    'PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU',
    'StandardGraphNet', 'AdaptiveGraphNet', 'PhysicsInformedGraphNet'
]
