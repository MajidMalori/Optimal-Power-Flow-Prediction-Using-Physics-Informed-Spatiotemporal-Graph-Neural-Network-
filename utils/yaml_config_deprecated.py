"""
DEPRECATED: This module has been merged into config.py

For backward compatibility, this module re-exports functions from config.py.
Please update your imports to use config.py directly:

OLD:
    from utils.yaml_config import merge_yaml_with_config
    
NEW:
    from config import _merge_yaml_with_config as merge_yaml_with_config

This file will be removed in a future version.
"""

import warnings
from config import _merge_yaml_with_config, _load_yaml_file, _flatten_dict, _convert_numeric_string

warnings.warn(
    "utils.yaml_config is deprecated and has been merged into config.py. "
    "Please update your imports to use config.py directly.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
merge_yaml_with_config = _merge_yaml_with_config
load_yaml_file = _load_yaml_file
flatten_dict = _flatten_dict
convert_numeric_string = _convert_numeric_string

__all__ = ['merge_yaml_with_config', 'load_yaml_file', 'flatten_dict', 'convert_numeric_string']

