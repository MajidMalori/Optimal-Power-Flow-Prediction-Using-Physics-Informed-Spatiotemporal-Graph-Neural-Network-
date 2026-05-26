"""Models and data module exports."""

from .data_module import PowerFlowDataModule


SPATIAL_MODELS = {"StandardGCN", "DynamicGCN", "PIGCN"}
RECURRENT_MODELS = {"PIGCLSTM", "PIGCGRU", "PIResnetGCLSTM", "PIResnetGCGRU"}


def get_model_registry():
    # Lazy imports to keep `import src.models` lightweight.
    from .gcn import StandardGCN
    from .dynamic_gcn import DynamicGCN
    from .pi_gcn import PIGCN
    from .pi_gclstm import PIGCLSTM
    from .pi_gcgru import PIGCGRU
    from .pi_resnet_gclstm import PIResnetGCLSTM
    from .pi_resnet_gcgru import PIResnetGCGRU

    return {
        "StandardGCN": StandardGCN,
        "DynamicGCN": DynamicGCN,
        "PIGCN": PIGCN,
        "PIGCLSTM": PIGCLSTM,
        "PIGCGRU": PIGCGRU,
        "PIResnetGCLSTM": PIResnetGCLSTM,
        "PIResnetGCGRU": PIResnetGCGRU,
    }


_EXPORTS = {
    "StandardGCN": (".gcn", "StandardGCN"),
    "DynamicGCN": (".dynamic_gcn", "DynamicGCN"),
    "PIGCN": (".pi_gcn", "PIGCN"),
    "PIGCLSTM": (".pi_gclstm", "PIGCLSTM"),
    "PIGCGRU": (".pi_gcgru", "PIGCGRU"),
    "PIResnetGCLSTM": (".pi_resnet_gclstm", "PIResnetGCLSTM"),
    "PIResnetGCGRU": (".pi_resnet_gcgru", "PIResnetGCGRU"),
    "PhysicsLoss": (".physics_loss", "PhysicsLoss"),
}


def __getattr__(name: str):
    if name in _EXPORTS:
        mod_name, attr = _EXPORTS[name]
        from importlib import import_module

        mod = import_module(mod_name, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PowerFlowDataModule",
    "SPATIAL_MODELS",
    "RECURRENT_MODELS",
    "get_model_registry",
    "StandardGCN",
    "DynamicGCN",
    "PIGCN",
    "PIGCLSTM",
    "PIGCGRU",
    "PIResnetGCLSTM",
    "PIResnetGCGRU",
    "PhysicsLoss",
]
