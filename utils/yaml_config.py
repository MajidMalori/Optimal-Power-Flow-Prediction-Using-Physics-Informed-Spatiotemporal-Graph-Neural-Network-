"""
YAML Configuration Loader for Physics-Informed Machine Learning Project

This module provides utilities to load configuration from YAML files and merge
them with the existing Config class, enabling version-controlled, reproducible
configuration management.

Usage:
    from utils.yaml_config import load_config_from_yaml, merge_yaml_with_config
    
    # Load YAML and create Config instance
    config = load_config_from_yaml('config.yaml', data_mode='test', ...)
    
    # Or merge YAML into existing Config instance
    config = Config(...)
    merge_yaml_with_config('config.yaml', config)
"""

import os
import yaml
import torch
from typing import Dict, Any, Optional
from pathlib import Path


def load_yaml_file(yaml_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        yaml_path: Path to YAML file (relative to project root or absolute)
    
    Returns:
        Dictionary containing YAML configuration
    
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML file is malformed
    """
    # Convert to Path object for easier handling
    yaml_path = Path(yaml_path)
    
    # If relative path, try to resolve relative to project root
    if not yaml_path.is_absolute():
        # Try to find project root (directory containing config.py)
        current_dir = Path(__file__).parent.parent  # utils/ -> project root
        yaml_path = current_dir / yaml_path
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        return {}
    
    return config_dict


def _convert_numeric_string(value: Any) -> Any:
    """
    Convert string representations of numbers to their proper numeric types.
    
    PyYAML sometimes parses scientific notation (e.g., "1e-6") as strings instead of floats.
    This function detects and converts such strings to their numeric equivalents.
    
    Args:
        value: Value to potentially convert
    
    Returns:
        Converted value (float/int if numeric string, original value otherwise)
    
    Examples:
        _convert_numeric_string("1e-6") -> 1e-6 (float)
        _convert_numeric_string("50") -> 50 (int)
        _convert_numeric_string("0.001") -> 0.001 (float)
        _convert_numeric_string("hello") -> "hello" (str, unchanged)
    """
    if not isinstance(value, str):
        return value
    
    # Try to convert to float first (handles scientific notation)
    try:
        float_val = float(value)
        # If it's an integer representation, return int
        if '.' not in value and 'e' not in value.lower() and 'E' not in value:
            try:
                return int(value)
            except ValueError:
                pass
        return float_val
    except (ValueError, OverflowError):
        # Not a numeric string, return as-is
        return value


def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Example:
        {'training': {'learning_rate': 0.001}} -> {'training_learning_rate': 0.001}
    
    Args:
        nested_dict: Nested dictionary to flatten
        parent_key: Parent key prefix (for recursion)
        sep: Separator between keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def set_nested_attr(obj: Any, attr_path: str, value: Any, sep: str = '_'):
    """
    Set a nested attribute on an object using dot notation or separator.
    
    Example:
        set_nested_attr(config, 'training_learning_rate', 0.001)
        # Sets config.training.learning_rate = 0.001 (if training is a dict)
        # Or config.TRAINING_LEARNING_RATE = 0.001 (if using flat structure)
    
    Args:
        obj: Object to set attribute on
        attr_path: Path to attribute (e.g., 'training_learning_rate' or 'TRAINING_LEARNING_RATE')
        value: Value to set
        sep: Separator used in attr_path
    """
    # Try to set as uppercase attribute first (Config class uses UPPERCASE)
    attr_name_upper = attr_path.upper().replace(sep, '_')
    if hasattr(obj, attr_name_upper):
        setattr(obj, attr_name_upper, value)
        return
    
    # Try to set as original case
    if hasattr(obj, attr_path):
        setattr(obj, attr_path, value)
        return
    
    # Try nested structure (e.g., training.learning_rate)
    parts = attr_path.split(sep)
    if len(parts) > 1:
        # Try to find nested object
        first_part = parts[0].upper()
        if hasattr(obj, first_part):
            nested_obj = getattr(obj, first_part)
            if isinstance(nested_obj, dict):
                nested_key = sep.join(parts[1:])
                nested_obj[nested_key] = value
                return
    
    # If all else fails, set as uppercase attribute (Config convention)
    setattr(obj, attr_name_upper, value)


def merge_yaml_with_config(yaml_path: str, config_obj: Any, verbose: bool = False) -> None:
    """
    Merge YAML configuration into an existing Config object.
    
    This function loads a YAML file and updates the Config object's attributes
    with values from the YAML file. It handles nested structures and converts
    them to the Config class's attribute naming convention (UPPERCASE).
    
    Args:
        yaml_path: Path to YAML configuration file
        config_obj: Config object to update
        verbose: If True, print which attributes are being set
    
    Example:
        config = Config(data_mode='test')
        merge_yaml_with_config('config.yaml', config)
        # Now config.LEARNING_RATE, config.NUM_EPOCHS, etc. are set from YAML
    """
    yaml_config = load_yaml_file(yaml_path)
    
    # Mapping from YAML keys to Config attribute names
    # This handles the conversion from nested YAML structure to Config's flat UPPERCASE attributes
    attribute_mapping = {
        # System configuration
        'system_device': 'DEVICE',
        'system_num_buses': 'NUM_BUSES',
        'system_seed': 'SEED',
        'system_num_workers': 'NUM_WORKERS',
        'system_case_name': 'CASE_NAME',
        
        # Training configuration
        'training_batch_size': 'BATCH_SIZE',
        'training_learning_rate': 'LEARNING_RATE',
        'training_max_grad_norm': 'MAX_GRAD_NORM',
        'training_num_epochs': 'NUM_EPOCHS',
        'training_early_stopping_patience': 'EARLY_STOPPING_PATIENCE',
        'training_use_learning_rate_scheduler': 'USE_LEARNING_RATE_SCHEDULER',
        'training_cosine_annealing_lr_t_max': 'COSINEANNEALINGLR_T_MAX',
        'training_cosine_annealing_lr_eta_min': 'COSINEANNEALINGLR_ETA_MIN',
        'training_weight_decay': 'WEIGHT_DECAY',
        
        # Physics configuration
        'physics_warmup_epochs': 'PHYSICS_WARMUP_EPOCHS',
        'physics_voltage_min': 'V_MIN',
        'physics_voltage_max': 'V_MAX',
        'physics_apparent_power_max': 'S_MAX',
        
        # Data configuration
        'physics_split_mode': 'DATA_SPLIT_MODE',
        'physics_splits_train': 'TRAIN_SPLIT',
        'physics_splits_val': 'VAL_SPLIT',
        'data_hours_per_day': 'HOURS_PER_DAY',
        'data_sequence_length': 'SEQUENCE_LENGTH',
        
        # MOOPF configuration
        'moopf_weights_loss': 'MOOPF_WEIGHT_LOSS',
        'moopf_weights_voltage_deviation': 'MOOPF_WEIGHT_VDEV',
        'moopf_weights_carbon': 'MOOPF_WEIGHT_CARBON',
        
        # Contingency configuration
        'contingency_enable': 'ENABLE_CONTINGENCY_ANALYSIS',
        'contingency_top_k': 'CONTINGENCY_TOP_K',
        'contingency_method': 'CONTINGENCY_METHOD',
        
        # Heteroscedastic configuration
        'heteroscedastic_softplus_beta': 'HETEROSCEDASTIC_SOFTPLUS_BETA',
        'heteroscedastic_use_clamping': 'HETEROSCEDASTIC_USE_CLAMPING',
        'heteroscedastic_weight_violations': 'HETEROSCEDASTIC_WEIGHT_VIOLATIONS',
        
        # Empirical Bayes configuration
        'empirical_bayes_enable': 'USE_EMPIRICAL_BAYES',
        'empirical_bayes_burn_in_epochs': 'EB_BURN_IN_EPOCHS',
        'empirical_bayes_update_frequency': 'EB_UPDATE_FREQUENCY',
        'empirical_bayes_hyperparameter_steps': 'EB_HYPERPARAMETER_STEPS',
        'empirical_bayes_hyperparameter_lr': 'EB_HYPERPARAMETER_LR',
    }
    
    # Flatten YAML config for easier processing
    flat_yaml = flatten_dict(yaml_config)
    
    # Apply mappings and set attributes
    for yaml_key, value in flat_yaml.items():
        # Skip None values (they indicate "use default")
        if value is None:
            continue
        
        # Convert numeric strings to proper types (PyYAML sometimes parses scientific notation as strings)
        value = _convert_numeric_string(value)
        
        # Get Config attribute name from mapping, or use uppercase version
        config_attr = attribute_mapping.get(yaml_key, yaml_key.upper().replace('-', '_'))
        
        # Handle special cases
        if yaml_key == 'system_device' and value == 'cuda':
            # Auto-detect CUDA availability
            if not torch.cuda.is_available():
                if verbose:
                    print(f"[YAML Config] CUDA requested but not available, using CPU")
                value = 'cpu'
        
        # Set attribute on config object (YAML is single source of truth - always set, even if doesn't exist)
        # Config uses class attributes, so we try instance first, then class
        if hasattr(config_obj, config_attr):
            old_value = getattr(config_obj, config_attr, None)
            setattr(config_obj, config_attr, value)
            if verbose:
                print(f"[YAML Config] Set {config_attr} = {value} (was {old_value})")
        elif hasattr(config_obj.__class__, config_attr):
            # Try to set on class (for class attributes)
            old_value = getattr(config_obj.__class__, config_attr, None)
            setattr(config_obj.__class__, config_attr, value)
            if verbose:
                print(f"[YAML Config] Set {config_attr} (class attribute) = {value} (was {old_value})")
        else:
            # Attribute doesn't exist - create it on the instance (YAML is source of truth)
            setattr(config_obj, config_attr, value)
            if verbose:
                print(f"[YAML Config] Created {config_attr} = {value} (new attribute from YAML)")
    
    # Handle model capacity settings (stored in Config class, not Args)
    if 'model_capacity_bus_33' in flat_yaml:
        from config import Config
        Config.CAPACITY_33_BUS = flat_yaml['model_capacity_bus_33']
        if verbose:
            print(f"[YAML Config] Set Config.CAPACITY_33_BUS = {flat_yaml['model_capacity_bus_33']}")
    
    if 'model_capacity_bus_57' in flat_yaml:
        from config import Config
        Config.CAPACITY_57_BUS = flat_yaml['model_capacity_bus_57']
        if verbose:
            print(f"[YAML Config] Set Config.CAPACITY_57_BUS = {flat_yaml['model_capacity_bus_57']}")
    
    if 'model_capacity_bus_118' in flat_yaml:
        from config import Config
        Config.CAPACITY_118_BUS = flat_yaml['model_capacity_bus_118']
        if verbose:
            print(f"[YAML Config] Set Config.CAPACITY_118_BUS = {flat_yaml['model_capacity_bus_118']}")
    
    # Handle experimental arguments (stored in Config class, not Args)
    experimental_mappings = {
        'experimental_test_config': 'test_config',
        'experimental_bus_systems': 'bus_systems',
        'experimental_models_to_train': 'models_to_train',
        'experimental_data_mode': 'data_mode',
        'experimental_train_timesteps': 'train_timesteps',
        'experimental_test_timesteps': 'test_timesteps',
        'experimental_plot_data_info': 'plot_data_info',

        'experimental_force_cpu': 'force_cpu',
        'experimental_parallel_data_loading': 'parallel_data_loading',
        'experimental_data_workers': 'data_workers',
        'experimental_save_results': 'save_results',
        'experimental_clear_results': 'clear_results',
    }
    
    for yaml_key, config_attr in experimental_mappings.items():
        if yaml_key in flat_yaml and flat_yaml[yaml_key] is not None:
            from config import Config
            setattr(Config, config_attr, flat_yaml[yaml_key])
            if verbose:
                print(f"[YAML Config] Set Config.{config_attr} = {flat_yaml[yaml_key]}")
    
    # Handle system-specific limits (voltage) based on CASE_NAME
    # CRITICAL: This must run AFTER CASE_NAME is set (if set via YAML)
    # system_limits in YAML: { case33: {v_min: 0.90, v_max: 1.10}, case57: {...}, case118: {...} }
    if 'system_limits' in yaml_config:
        case_name = getattr(config_obj, 'CASE_NAME', None)
        if case_name:
            case_name_lower = case_name.lower()
            system_limits = yaml_config['system_limits']
            
            if case_name_lower in system_limits:
                limits = system_limits[case_name_lower]
                if 'v_min' in limits:
                    setattr(config_obj, 'V_MIN', limits['v_min'])
                    if verbose:
                        print(f"[YAML Config] Set V_MIN = {limits['v_min']} for {case_name}")
                if 'v_max' in limits:
                    setattr(config_obj, 'V_MAX', limits['v_max'])
                    if verbose:
                        print(f"[YAML Config] Set V_MAX = {limits['v_max']} for {case_name}")
            else:
                # Case name not found in system_limits - warn but don't fail
                available_cases = list(system_limits.keys())
                if verbose:
                    print(f"[YAML Config] WARNING: {case_name_lower} not found in system_limits. Available: {available_cases}")
        else:
            # CASE_NAME not set yet - this is expected on initial Config() creation
            # The limits will be set later when CASE_NAME is set during training
            if verbose:
                print(f"[YAML Config] NOTE: CASE_NAME not set yet, skipping system-specific voltage limits")



def load_config_from_yaml(yaml_path: str = 'config.yaml', **kwargs) -> Any:
    """
    Load configuration from YAML file and create a Config instance.
    
    This is a convenience function that loads YAML and creates a Config object
    with the YAML values applied. Any keyword arguments override YAML values.
    
    Args:
        yaml_path: Path to YAML configuration file
        **kwargs: Additional arguments to pass to Config.__init__()
    
    Returns:
        Config instance with YAML values applied
    
    Example:
        config = load_config_from_yaml('config.yaml', data_mode='test')
    """
    from config import Config
    
    # Create Config instance with kwargs
    config = Config(**kwargs)
    
    # Merge YAML configuration
    if os.path.exists(yaml_path) or os.path.exists(os.path.join(os.path.dirname(__file__), '..', yaml_path)):
        merge_yaml_with_config(yaml_path, config, verbose=False)
    
    return config


def save_config_to_yaml(config_obj: Any, yaml_path: str = 'config_generated.yaml') -> None:
    """
    Save a Config object's attributes to a YAML file.
    
    This is useful for creating a YAML file from an existing Config instance,
    or for saving the current configuration state.
    
    Args:
        config_obj: Config object to save
        yaml_path: Path to save YAML file to
    
    Example:
        config = Config(data_mode='test')
        save_config_to_yaml(config, 'my_config.yaml')
    """
    # Collect all uppercase attributes (Config convention)
    config_dict = {}
    
    # System configuration
    config_dict['system'] = {
        'device': getattr(config_obj, 'DEVICE', 'cpu'),
        'num_buses': getattr(config_obj, 'NUM_BUSES', [33, 57, 118]),
        'seed': getattr(config_obj, 'SEED', 42),
        'num_workers': getattr(config_obj, 'NUM_WORKERS', 2),
    }
    
    # Training configuration
    config_dict['training'] = {
        'batch_size': getattr(config_obj, 'BATCH_SIZE', 64),
        'learning_rate': getattr(config_obj, 'LEARNING_RATE', 0.0005),
        'max_grad_norm': getattr(config_obj, 'MAX_GRAD_NORM', 1.0),
        'num_epochs': getattr(config_obj, 'NUM_EPOCHS', 50),
        'early_stopping_patience': getattr(config_obj, 'EARLY_STOPPING_PATIENCE', 10),
        'use_learning_rate_scheduler': getattr(config_obj, 'USE_LEARNING_RATE_SCHEDULER', True),
        'cosine_annealing_lr': {
            't_max': getattr(config_obj, 'COSINEANNEALINGLR_T_MAX', None),
            'eta_min': getattr(config_obj, 'COSINEANNEALINGLR_ETA_MIN', 1e-6),
        },
        'weight_decay': getattr(config_obj, 'WEIGHT_DECAY', 0.0),
    }
    
    # Physics configuration
    config_dict['physics'] = {
        'warmup_epochs': getattr(config_obj, 'PHYSICS_WARMUP_EPOCHS', 10),
        'voltage': {
            'min': getattr(config_obj, 'V_MIN', 0.90),
            'max': getattr(config_obj, 'V_MAX', 1.10),
        },
        'apparent_power_max': getattr(config_obj, 'S_MAX', 1.2),
    }
    
    # Data configuration
    config_dict['data'] = {
        'split_mode': getattr(config_obj, 'DATA_SPLIT_MODE', 'blocked_timeseries'),
        'splits': {
            'train': getattr(config_obj, 'TRAIN_SPLIT', 0.6),
            'val': getattr(config_obj, 'VAL_SPLIT', 0.2),
        },
        'hours_per_day': getattr(config_obj, 'HOURS_PER_DAY', 24),
        'sequence_length': getattr(config_obj, 'SEQUENCE_LENGTH', 5),
    }
    
    # MOOPF configuration
    config_dict['moopf'] = {
        'weights': {
            'loss': getattr(config_obj, 'MOOPF_WEIGHT_LOSS', 1/3),
            'voltage_deviation': getattr(config_obj, 'MOOPF_WEIGHT_VDEV', 1/3),
            'carbon': getattr(config_obj, 'MOOPF_WEIGHT_CARBON', 1/3),
        }
    }
    
    # Contingency configuration
    config_dict['contingency'] = {
        'enable': getattr(config_obj, 'ENABLE_CONTINGENCY_ANALYSIS', True),
        'top_k': getattr(config_obj, 'CONTINGENCY_TOP_K', 10),
        'method': getattr(config_obj, 'CONTINGENCY_METHOD', 'power_flow'),
    }
    
    # Heteroscedastic configuration
    config_dict['heteroscedastic'] = {
        'softplus_beta': getattr(config_obj, 'HETEROSCEDASTIC_SOFTPLUS_BETA', 1.0),
        'use_clamping': getattr(config_obj, 'HETEROSCEDASTIC_USE_CLAMPING', False),
        'weight_violations': getattr(config_obj, 'HETEROSCEDASTIC_WEIGHT_VIOLATIONS', True),
    }
    
    # Empirical Bayes configuration
    config_dict['empirical_bayes'] = {
        'enable': getattr(config_obj, 'USE_EMPIRICAL_BAYES', True),
        'burn_in_epochs': getattr(config_obj, 'EB_BURN_IN_EPOCHS', 2),
        'update_frequency': getattr(config_obj, 'EB_UPDATE_FREQUENCY', 5),
        'hyperparameter_steps': getattr(config_obj, 'EB_HYPERPARAMETER_STEPS', 20),
        'hyperparameter_lr': getattr(config_obj, 'EB_HYPERPARAMETER_LR', 0.001),
    }
    
    # Write to YAML file
    yaml_path = Path(yaml_path)
    if not yaml_path.is_absolute():
        yaml_path = Path(__file__).parent.parent / yaml_path
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"[YAML Config] Saved configuration to {yaml_path}")

