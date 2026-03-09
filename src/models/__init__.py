from .gcn import StandardGCN
from .dynamic_gcn import DynamicGCN
from .pi_gcn import PIGCN
from .pi_gclstm import PIGCLSTM
from .pi_gcgru import PIGCGRU
from .pi_resnet_gclstm import PIResnetGCLSTM
from .pi_resnet_gcgru import PIResnetGCGRU
from .data_module import PowerFlowDataModule
from .physics_loss import PhysicsLoss

# Registry for all available architectures
MODEL_REGISTRY = {
    "StandardGCN": StandardGCN,
    "DynamicGCN": DynamicGCN,
    "PIGCN": PIGCN,
    "PIGCLSTM": PIGCLSTM,
    "PIGCGRU": PIGCGRU,
    "PIResnetGCLSTM": PIResnetGCLSTM,
    "PIResnetGCGRU": PIResnetGCGRU
}

SPATIAL_MODELS = {"StandardGCN", "DynamicGCN", "PIGCN"}
RECURRENT_MODELS = {"PIGCLSTM", "PIGCGRU", "PIResnetGCLSTM", "PIResnetGCGRU"}

__all__ = [
    "StandardGCN",
    "DynamicGCN",
    "PIGCN",
    "PIGCLSTM",
    "PIGCGRU",
    "PIResnetGCLSTM",
    "PIResnetGCGRU",
    "PowerFlowDataModule",
    "PhysicsLoss",
    "MODEL_REGISTRY",
    "SPATIAL_MODELS",
    "RECURRENT_MODELS"
]
