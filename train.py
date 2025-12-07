import os
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import copy
import sys
import yaml
import signal
import gc
import random

# Global data cache for bus systems (used across multiple functions)
_file_metadata = None
_adjacency = None
_ybus_metadata = None
_normalizer = None
_topology_cache = None
_topology_ids = None

def check_gpu_memory():
    """Check available GPU memory and return status"""
    if not torch.cuda.is_available():
        return {'available': False, 'total': 0, 'free': 0, 'used': 0}
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_reserved()
    free_memory = total_memory - allocated_memory
    
    return {
        'available': True,
        'total': total_memory,
        'allocated': allocated_memory,
        'cached': cached_memory,
        'free': free_memory
    }

def clear_gpu_memory():
    """
    Clear GPU memory cache when swapping models.
    Only called between different model architectures to prevent OOM.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_adaptive_batch_size(num_buses, base_batch_size=32):
    """
    Get batch size based on system size.
    Larger systems use smaller batches due to memory constraints.
    """
    if num_buses <= 33:
        return base_batch_size
    elif num_buses <= 69:
        return max(16, base_batch_size // 2)
    elif num_buses <= 118:
        return max(8, base_batch_size // 4)
    else:
        return max(4, base_batch_size // 8)

def cleanup_bus_system_data():
    """Clean up data between bus systems to free memory."""
    global _file_metadata, _adjacency, _ybus_metadata, _normalizer, _topology_cache, _topology_ids
    
    # Reset global variables to None (Python's GC will handle cleanup)
    _file_metadata = None
    _adjacency = None
    _ybus_metadata = None
    _normalizer = None
    _topology_cache = None
    _topology_ids = None

def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for memory efficiency (currently unused)."""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled for memory efficiency")

def _auto_detect_num_workers(device):
    """
    Automatically determine optimal number of data loading workers based on system specs.
    
    Factors considered:
    - Windows: Must use 0 (multiprocessing issues)
    - RAM: Low RAM (< 16GB) = fewer workers to avoid memory pressure
    - GPU: GPU available = can use more workers (GPU handles computation)
    - CPU cores: More cores = can support more workers
    - Best practices: Balance between parallelism and overhead
    
    Args:
        device: torch.device (cpu or cuda)
        
    Returns:
        int: Optimal number of workers (0-8)
    """
    import sys
    
    # Windows: Must use 0 due to multiprocessing limitations
    if sys.platform == 'win32':
        return 0
    
    try:
        # Get CPU cores
        cpu_cores = os.cpu_count() or 4  # Fallback to 4 if None
        
        # Try to get RAM (optional dependency)
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback: Assume moderate RAM if psutil not available
            ram_gb = 16.0
        
        # Check GPU availability
        has_gpu = device.type == 'cuda' and torch.cuda.is_available()
        
        # Decision logic:
        # 1. Low RAM (< 8GB): Conservative (2 workers max)
        # 2. Moderate RAM (8-16GB): Moderate (2-4 workers)
        # 3. High RAM (> 16GB): Can use more workers
        
        if ram_gb < 8.0:
            # Very low RAM: Use minimal workers
            num_workers = min(2, cpu_cores // 4)
        elif ram_gb < 16.0:
            # Low-moderate RAM (like user's 8GB system)
            if has_gpu:
                # GPU available: Can use moderate workers (GPU handles computation)
                num_workers = min(4, cpu_cores // 2)
            else:
                # CPU-only: Use fewer workers to avoid memory pressure
                num_workers = min(2, cpu_cores // 4)
        else:
            # High RAM: Can use more workers
            if has_gpu:
                # GPU available: Use more workers (GPU handles computation, RAM available)
                num_workers = min(8, cpu_cores // 2)
            else:
                # CPU-only: Moderate workers (more RAM but CPU handles computation)
                num_workers = min(4, cpu_cores // 2)
        
        # Ensure at least 1 worker if we have resources (unless Windows)
        num_workers = max(1, num_workers) if cpu_cores >= 2 else 0
        
        return num_workers
        
    except Exception as e:
        # Fallback: Conservative default
        print(f"Warning: Could not auto-detect num_workers: {e}. Using default: 2")
        return 2

def get_safe_device(force_cpu=False, min_free_memory_gb=2.0):
    """Get a safe device for training, with automatic fallback to CPU if GPU memory is insufficient"""
    if force_cpu:
        return torch.device('cpu'), 'forced_cpu'
    
    if not torch.cuda.is_available():
        return torch.device('cpu'), 'no_cuda'
    
    # Check available GPU memory
    memory_info = check_gpu_memory()
    if not memory_info['available']:
        return torch.device('cpu'), 'no_cuda'
    
    free_memory_gb = memory_info['free'] / (1024**3)
    total_memory_gb = memory_info['total'] / (1024**3)
    
    print(f"GPU Memory: {free_memory_gb:.2f} GB free / {total_memory_gb:.2f} GB total")
    
    # If free memory is less than minimum required, fallback to CPU
    if free_memory_gb < min_free_memory_gb:
        print(f"Warning: GPU memory insufficient ({free_memory_gb:.2f} GB < {min_free_memory_gb} GB), falling back to CPU")
        return torch.device('cpu'), 'insufficient_gpu_memory'
    
    return torch.device('cuda'), 'gpu_available'

# Set matplotlib backend before any plotting imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent threading issues

from utils.data_loader import load_power_system_data, create_data_loaders
from utils.metrics import PowerSystemLoss
from utils.data_validation import validate_data_before_training
from utils.optimization import (mosoa_optimizer, setup_hyperparameter_bounds, create_model_kwargs, 
                               generate_run_name, process_optimization_params, format_params_concise,
                               calculate_objective_score)
from utils.uncertainty_analysis import generate_uncertainty_visualizations
from utils.evaluation_plots import (plot_predicted_vs_actual, plot_error_distributions, plot_calibration_diagram)
from utils.visualization import plot_training_history, plot_convergence, plot_all_renewable_impacts
from utils.evaluation import (evaluate_performance,
                             evaluate_renewable_impacts_from_predictions,
                             evaluate_model_with_uncertainty,
                             evaluate_moopf_objectives_normalized,
                             save_results)
from utils.evaluation_summary_funcs import (print_model_summary,
                                           save_best_model_results,
                                           save_model_results_csv,
                                           print_comprehensive_summary)
from trainers.model_trainer import PowerSystemTrainer
from utils.forensic_logger import init_forensic_logger, close_logger
from utils.shutdown_flag import set_shutdown
from config import Config
from tqdm import tqdm


def setup_logging(log_path: str):
    """Initializes logging to both file and console."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_dir = os.path.dirname(log_path)
    if log_dir: 
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()]
    )




# Global variable to store config for signal handler
_config_instance = None

def signal_handler(signum, _):
    """Handle interrupt signals gracefully - set flag instead of printing directly."""
    set_shutdown()

def setup_professional_logging():
    """Setup professional logging with memory tracking"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def main():
    global _config_instance
    
    # Setup professional logging
    logger = setup_professional_logging()
    logger.info("Starting training session with memory optimizations")

        
    # Clean up old debug logs if forensic logging is enabled
    if os.environ.get('FORENSIC_DEBUG', 'false').lower() == 'true':
        debug_logs_dir = os.path.join(os.path.dirname(__file__), 'debug_logs')
        if os.path.exists(debug_logs_dir):
            try:
                import shutil
                shutil.rmtree(debug_logs_dir)
                logger.info(f"Cleaned up old debug logs: {debug_logs_dir}")
            except Exception as e:
                logger.warning(f"Could not delete debug logs: {e}")
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Set PyTorch CUDA memory allocation configuration to prevent fragmentation
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent memory fragmentation")
    
    # Parse command line arguments for overrides
    parser = argparse.ArgumentParser(description='Physics-Informed Model Training')
    parser.add_argument('--time_steps', type=int, default=None, help='Override number of time steps')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    parser.add_argument('--mode', type=str, default=None, choices=['train', 'test'], help='Data mode (train/test)')
    args = parser.parse_args()
    
    # Determine data mode with correct priority: CLI > YAML > Default ('train')
    if args.mode is not None:
        data_mode_to_use = args.mode
    else:
        # Load from YAML if no CLI arg
        try:
            yaml_path = args.config if os.path.isabs(args.config) else os.path.join(os.path.dirname(__file__), args.config)
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                data_mode_to_use = yaml_config.get('experimental', {}).get('data_mode', 'train')
        except:
            data_mode_to_use = 'train'
    
    # Pass CLI overrides to Config (CLI args override YAML)
    # Note: YAML is loaded inside Config.__init__, so we pass defaults and let Config read from YAML
    # YAML values will override these defaults inside Config.__init__
    base_config = Config(
        yaml_config_path=args.config,
        load_yaml=True,
        data_mode=data_mode_to_use,
        train_timesteps=args.time_steps,  # None if not provided, Config will read from YAML
        test_timesteps=args.time_steps,   # None if not provided, Config will read from YAML
        clear_results=False,  # Default, will be overridden by YAML if present
        hours_per_day=24,  # Standard value
        sequence_length=5   # Standard value
    )
    
    # Explicitly create run directories for training
    base_config.create_run_directories()
    
    _config_instance = base_config  # Store for signal handler
    
    # Parse bus systems to test (from Config, not Args)
    def parse_bus_systems(bus_systems_arg):
        """Parse bus systems argument and return list of bus numbers to test."""
        # Handle list (from YAML)
        if isinstance(bus_systems_arg, list):
            return [int(b) for b in bus_systems_arg]
        
        # Handle string
        if isinstance(bus_systems_arg, str):
            if bus_systems_arg.lower() == 'all':
                return base_config.NUM_BUSES
            else:
                # Parse comma-separated values
                bus_list = []
                for bus_str in bus_systems_arg.split(','):
                    bus_str = bus_str.strip()
                    try:
                        bus_num = int(bus_str)
                        if bus_num in base_config.NUM_BUSES:
                            bus_list.append(bus_num)
                        else:
                            print(f"WARNING: {bus_num}-bus system not available. Available: {base_config.NUM_BUSES}")
                    except ValueError:
                        print(f"WARNING: Invalid bus system '{bus_str}'. Skipping.")
                return bus_list if bus_list else base_config.NUM_BUSES
        
        # Handle single integer
        if isinstance(bus_systems_arg, int):
            return [bus_systems_arg]
        
        # Fallback
        return base_config.NUM_BUSES
    
    bus_systems_to_test = parse_bus_systems(getattr(Config, 'bus_systems', 'all'))
    
    # Track all results for comprehensive summary
    all_results = []
    
    # STEP 1: Concise run information (one line)
    run_info = base_config.get_run_info()
    data_mode = getattr(Config, 'data_mode', 'test')
    
    # STEP 2: Validate data before training
    if not validate_data_before_training(base_config, bus_systems_to_test):
        raise RuntimeError(
            "Data validation failed! Required data files are missing or invalid.\n"
            "Run data generation first: python data/main.py"
        )
    
    # Get seed from config (loaded from YAML) for reproducibility
    seed = getattr(base_config, 'SEED', 42)
    
    # IMPORTANT: Set environment variables BEFORE any other operations for full reproducibility
    # These must be set before PyTorch operations to ensure determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA deterministic operations
    
    # Set all random seeds for full reproducibility
    print(f"\nSetting random seed: {seed} (for reproducibility)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Enable deterministic mode for PyTorch (slower but fully reproducible)
    # Note: This may reduce performance but ensures exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms in PyTorch (if available)
    # This ensures operations like matrix multiplication are deterministic
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        # Older PyTorch versions (<1.8) don't have this function
        pass
    
    # === DEVICE AND PARALLEL CONFIGURATION ===
    force_cpu = getattr(Config, 'force_cpu', False)
    device, _ = get_safe_device(force_cpu, min_free_memory_gb=2.0)
    
    # Store Config attributes for later use (replaces args object)
    # Use instance values (YAML has been loaded by now)
    _config_attrs = {

        'clear_results': getattr(Config, 'clear_results', False),
    }
    clear_gpu_memory()
    
    # Auto-configure NUM_WORKERS based on system specs (CPU, RAM, GPU, OS)
    data_workers_config = getattr(Config, 'data_workers', 'auto')
    if data_workers_config == 'auto':
        base_config.NUM_WORKERS = _auto_detect_num_workers(device)
        print(f"Auto-detected data workers: {base_config.NUM_WORKERS} (based on system specs)")
    else:
        base_config.NUM_WORKERS = int(data_workers_config)
        print(f"Using configured data workers: {base_config.NUM_WORKERS}")
    
    # Ensure parallel_data_loading setting is respected
    parallel_data_loading = getattr(Config, 'parallel_data_loading', True)
    if not parallel_data_loading:
        base_config.NUM_WORKERS = 0
    
    # Get GPU info for startup printout
    if device.type == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device_info = f"CUDA ({gpu_name}, {gpu_memory_gb:.1f}GB)"
    else:
        device_info = "CPU"
    
    # Single concise line with all startup info (enhanced with GPU details)
    test_config = getattr(Config, 'test_config', 'all')
    print(f"RUN: {run_info['run_id']} | Config: {test_config} | Mode: {data_mode.upper()} | Buses: {bus_systems_to_test} | Device: {device_info} | Workers: {base_config.NUM_WORKERS}")

    # Get model configurations from config
    model_class_map = base_config.get_model_class_map()
    model_config_map = base_config.model_config_map
    
    # Hierarchical model selection:
    # 1. If models_to_train is specified and not "all" → Use those (override test_config)
    # 2. Otherwise → Use test_config
    models_to_train = getattr(Config, 'models_to_train', None)
    
    if models_to_train and models_to_train != "all":
        # User specified exact models → Override test_config
        if isinstance(models_to_train, str):
            models_to_test = [m.strip() for m in models_to_train.split(',')]
        elif isinstance(models_to_train, list):
            models_to_test = models_to_train
        else:
            models_to_test = [models_to_train]
        print(f"Using explicit model selection (overriding test_config): {models_to_test}")
    else:
        # Use test_config presets
        models_to_test = base_config.get_models_to_test(test_config)
        print(f"Using test_config '{test_config}': {models_to_test}")
    
    if not models_to_test:
        raise ValueError(
            f"No models selected for training!\n"
            f"Available test_config options: quick, core, comprehensive, physics_only, non_physics_only, sequential_only, all\n"
            f"Or set models_to_train to specific models like: ['GCN', 'adaptiveGCN']"
        )

    # === MAIN TRAINING EXECUTION ===
    for num_buses in bus_systems_to_test:
        # Get adaptive MoSOA parameters for this system size
        mosoa_params = base_config._ModelConfig.get_adaptive_mosoa_params(num_buses)
        print(f"{'='*80}\n{num_buses}-BUS | MoSOA: {mosoa_params['num_seagulls']} seagulls x {mosoa_params['max_iterations']} iters ({mosoa_params['strategy']})\n{'='*80}")
        
        # Initialize data collectors for comparative plots
        bus_renewable_data = {}  # model_name -> renewable_impact_dataframe
        bus_convergence_data = {}  # model_name -> convergence_history
        all_tested_models = []  # Track all models tested (including non-physics)
        
        case_name = f"case{num_buses}"
        # Set case name in config to enable system-specific base power determination
        base_config.CASE_NAME = case_name
        
        # Load system-specific voltage limits from YAML config
        # This must be done AFTER setting CASE_NAME
        config_yaml_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        try:
            with open(config_yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            if 'system_limits' in yaml_config:
                case_name_lower = case_name.lower()
                system_limits = yaml_config['system_limits']
                if case_name_lower in system_limits:
                    limits = system_limits[case_name_lower]
                    if 'base_mva' in limits:
                        base_config.BASE_MVA = limits['base_mva']
                    if 'v_min' in limits:
                        base_config.V_MIN = limits['v_min']
                    if 'v_max' in limits:
                        base_config.V_MAX = limits['v_max']
        except (FileNotFoundError, yaml.YAMLError) as e:
            raise RuntimeError(f"Failed to load voltage limits from config.yaml: {e}. Ensure config.yaml exists and is properly formatted.")
        
        try:
            data_tuple = load_power_system_data(base_config, case_name)
            # Unpack: (file_metadata, base_adjacency, ybus_metadata, normalizer, topology_cache, topology_ids)
            # Unpack data tuple (6 items expected)
            _file_metadata, _adjacency, _ybus_metadata, _normalizer, _topology_cache, _topology_ids = data_tuple
            
            bus_models_to_test = models_to_test.copy()
        except FileNotFoundError as e:
            print(f"WARNING: Skipping {num_buses}-bus system - data not found: {e}")
            print(f"  Run data generation for this system: python data/main.py")
            continue

        for model_name in bus_models_to_test:
            print(f"\n{'='*60}\n{model_name} on {num_buses}-bus\n{'='*60}")
            
            model_specific_results = []
            model_config = model_config_map[model_name]

            # Get model characteristics from config
            is_sequential = base_config.is_sequential_model(model_name)
            is_physics_informed = base_config.is_physics_informed(model_name)
            uses_adaptive_graph = base_config.uses_adaptive_graph(model_name)

            # Setup hyperparameter bounds
            param_bounds = setup_hyperparameter_bounds(
                model_name, model_config, num_buses, 
                is_physics_informed, is_sequential, uses_adaptive_graph
            )

            param_keys = list(param_bounds.keys())
            dim = len(param_bounds)
            lower_bounds = [b[0] for b in param_bounds.values()]
            upper_bounds = [b[1] for b in param_bounds.values()]

            # Track MoSOA iteration and run for display
            mosoa_iter = [0]  # Current iteration (1-indexed)
            mosoa_max_iter = mosoa_params['max_iterations']
            mosoa_run_total = [0]  # Total run counter across all iterations
            mosoa_runs_per_iter = mosoa_params['num_seagulls']

            def objective_function(params_array):
                params = process_optimization_params(param_keys, params_array)

                run_config = copy.deepcopy(base_config)
                for key, value in params.items(): 
                    setattr(run_config, key.upper(), value)
                run_config.NUM_BUSES = num_buses

                run_name = generate_run_name(model_name, params, num_buses, is_sequential)

                try:
                    # Clear GPU memory before starting
                    clear_gpu_memory()
                    
                    # Use adaptive batch size based on system size
                    run_config.BATCH_SIZE = get_adaptive_batch_size(
                        num_buses, 
                        run_config.BATCH_SIZE
                    )
                    
                    # Safety check: Cap batch size to prevent OOM
                    # For 118-bus systems, limit to 128 to prevent memory issues
                    if num_buses >= 118:
                        max_safe_batch = 128  # Conservative limit for 118-bus without accumulation
                        if run_config.BATCH_SIZE > max_safe_batch:
                            print(f"  Warning: Batch size {run_config.BATCH_SIZE} may cause OOM for {num_buses}-bus. Capping to {max_safe_batch}")
                            run_config.BATCH_SIZE = max_safe_batch
                    
                    
                    loaders = create_data_loaders(
                        _file_metadata, _adjacency, _ybus_metadata, _normalizer, base_config, 
                        is_static=(not is_sequential), topology_cache=_topology_cache, topology_ids=_topology_ids
                    )
                    train_loader, val_loader, test_loader = loaders

                    # Create model with optimized parameters (always OPF mode)
                    model_kwargs = create_model_kwargs(
                        model_config, params, num_buses, is_sequential, uses_adaptive_graph, 
                        model_name=model_name, config=run_config, normalizer=_normalizer,
                        is_physics_informed=is_physics_informed
                    )
                    
                    # Check memory before model creation
                    if torch.cuda.is_available():
                        memory_info = check_gpu_memory()
                        if memory_info['free'] < 1024**3:  # Less than 1GB free
                            print("Warning: Low GPU memory, clearing cache...")
                            clear_gpu_memory()
                    
                    # Create model with error handling for OOM
                    try:
                        model = model_class_map[model_name](**model_kwargs).to(device)
                        # Enable gradient checkpointing for large models
                        enable_gradient_checkpointing(model)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"CUDA OOM during model creation: {e}")
                            clear_gpu_memory()
                            # Try with smaller batch size
                            run_config.BATCH_SIZE = max(1, run_config.BATCH_SIZE // 2)
                            print(f"Retrying with reduced batch size: {run_config.BATCH_SIZE}")
                            # Recreate loaders with smaller batch size
                            loaders = create_data_loaders(
                                _file_metadata, _adjacency, _ybus_metadata, _normalizer, run_config, 
                                is_static=(not is_sequential), topology_cache=_topology_cache, topology_ids=_topology_ids
                            )
                            train_loader, val_loader, test_loader = loaders
                            model = model_class_map[model_name](**model_kwargs).to(device)
                            # Apply collapse prevention initialization
                        else:
                            raise e
                    
                    # Use appropriate loss function based on whether model is physics-informed
                    criterion = PowerSystemLoss(
                        config=run_config, 
                        normalizer=_normalizer, 
                        is_gcn=(not is_physics_informed)
                    ).to(device)
                    
                    # Golden Configuration: Use AdamW optimizer
                    # AdamW decouples weight decay from gradient updates (better generalization than Adam)
                    learning_rate = run_config.LEARNING_RATE
                    weight_decay = getattr(run_config, 'WEIGHT_DECAY', 0.0001)  # L2 regularization from config
                    # Combine model and criterion parameters for optimization
                    all_params = list(model.parameters()) + list(criterion.parameters())
                    optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)

                    # Initialize forensic logger if debug mode is enabled
                    debug_config = getattr(run_config, 'DEBUG_ENABLE', True)  # From config.yaml debug.enable
                    if debug_config:
                        # Get log_interval from config (default to 10 if not set)
                        log_interval = getattr(run_config, 'DEBUG_LOG_INTERVAL', 10)
                        
                        forensic_logger = init_forensic_logger(
                            log_dir="debug_logs",
                            model_name=model_name,
                            bus_system=str(num_buses),
                            enabled=True,
                            log_interval=log_interval
                        )
                        # Attach logger to model if it's a ForensicGCN
                        if hasattr(model, 'set_logger'):
                            model.set_logger(forensic_logger)

                    trainer = PowerSystemTrainer(model, criterion, optimizer, run_config, device, is_physics_informed)
                    
                    # Prepare configuration parameters for logging
                    config_params = {
                        'learning_rate': learning_rate,
                        'batch_size': run_config.BATCH_SIZE,
                        'gradient_accumulation_steps': getattr(run_config, 'GRADIENT_ACCUMULATION_STEPS', 1),
                        'num_epochs': run_config.NUM_EPOCHS,
                        'early_stopping_patience': getattr(run_config, 'EARLY_STOPPING_PATIENCE', 10),
                    }
                    # Add model-specific parameters
                    for key, value in params.items():
                        config_params[key] = value
                    
                    config_str = ", ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                                           for k, v in sorted(config_params.items())])
                    # Add MoSOA progress to config line (get from objective function attributes)
                    if hasattr(objective_function, '_current_iter'):
                        iter_info = f" [MoSOA {objective_function._current_iter}/{objective_function._max_iter}, run {objective_function._current_run}/{objective_function._runs_per_iter}]"
                    else:
                        iter_info = f" [MoSOA {mosoa_iter[0]}/{mosoa_max_iter}]"
                    print(f"  Config: {config_str}{iter_info}")
                    
                    # Train the model 
                    trainer.train(train_loader, val_loader, model_name=model_name, num_buses=num_buses, config_params=config_params)

                    # Get training history with validation total loss (includes physics)
                    training_history = trainer.get_training_history()
                    
                    # For physics-informed models: Use the TOTAL LOSS (MSE + Physics + Safety)
                    # For non-physics models: Use MSE only
                    if is_physics_informed and 'val_total_loss' in training_history:
                        # Use the final validation total loss (last epoch)
                        # This includes MSE + weighted physics + weighted safety with Kendall balancing
                        final_val_loss = training_history['val_total_loss'][-1]
                        final_val_loss = final_val_loss.item() if hasattr(final_val_loss, 'item') else float(final_val_loss)
                        
                        val_metrics = {
                            'total_loss': float(final_val_loss),
                            'mse': float(training_history['val_mse'][-1])
                        }
                    else:
                        # Non-physics: Just use MSE
                        val_metrics = evaluate_performance(model, val_loader, device, run_config, _normalizer, is_sequential, return_denormalized=False)
                    
                    # Get test metrics for final evaluation (use normalized MSE for consistency)
                    test_metrics = evaluate_performance(model, test_loader, device, run_config, _normalizer, is_sequential, return_denormalized=False)

                    # Calculate objective score for MoSOA optimization
                    total_loss = calculate_objective_score(val_metrics, run_config, is_physics_informed)

                    # Store the training history with the results
                    # Only store model state if it's not too large (to prevent memory issues)
                    model_state = None
                    try:
                        model_state = model.state_dict()
                        # Check if model state is too large (> 100MB)
                        state_size = sum(p.numel() * p.element_size() for p in model_state.values())
                        if state_size > 100 * 1024 * 1024:  # 100MB
                            print(f"  Model state too large ({state_size / 1024**2:.1f} MB), not storing for memory efficiency")
                            model_state = None
                    except Exception as e:
                        print(f"  Could not save model state: {e}")
                        model_state = None
                    
                    # Extract final physics_loss and safety_loss from training history
                    training_history = trainer.get_training_history()
                    final_physics_loss = training_history['val_physics_loss'][-1]
                    final_safety_loss = training_history['val_safety_loss'][-1]
                    
                    run_results = {
                        'run_name': run_name, 
                        'model_name': model_name, 
                        **params, 
                        **test_metrics,  # Final test performance for reporting
                        'val_metrics': val_metrics,  # Validation metrics used for optimization
                        'total_loss': total_loss,  # Based on validation metrics
                        'training_mse': val_metrics['mse'],
                        'physics_loss': final_physics_loss,  # Final validation physics loss
                        'safety_loss': final_safety_loss,  # Final validation safety loss
                        'training_history': training_history,
                        'model_state': model_state,  # May be None for large models
                        'model_config': run_config  
                    }
                    model_specific_results.append(run_results)

                    return total_loss
                    
                except Exception as e:
                    # Sanitize error message for Windows encoding compatibility
                    error_msg = str(e).replace('η', 'eta').replace('δ', 'delta').replace('σ', 'sigma').replace('λ', 'lambda')
                    logging.error(f"Run {run_name} failed: {error_msg}", exc_info=True)
                    return float('inf')

            # Always use MoSOA for hyperparameter optimization
            if True:  # MoSOA always enabled
                print(f"Optimizing with MoSOA: {mosoa_params['num_seagulls']} seagulls × {mosoa_params['max_iterations']} iterations")
                
                # Wrap objective function to track iteration and run numbers
                def objective_with_tracking(params_array):
                    # Update total run counter
                    mosoa_run_total[0] += 1
                    # Calculate iteration number (1-indexed)
                    mosoa_iter[0] = (mosoa_run_total[0] - 1) // mosoa_runs_per_iter + 1
                    # Calculate run number within current iteration (1-indexed)
                    mosoa_run_in_iter = ((mosoa_run_total[0] - 1) % mosoa_runs_per_iter) + 1
                    # Store for display in config line
                    objective_function._current_iter = mosoa_iter[0]
                    objective_function._current_run = mosoa_run_in_iter
                    objective_function._max_iter = mosoa_max_iter
                    objective_function._runs_per_iter = mosoa_runs_per_iter
                    return objective_function(params_array)
                
                best_score, best_position, history, iteration_details = mosoa_optimizer(
                    mosoa_params['num_seagulls'], 
                    mosoa_params['max_iterations'], 
                    lower_bounds, upper_bounds, dim, objective_with_tracking,
                    param_keys=param_keys
                )
            # MoSOA is the only optimization method (trial_based_search removed)

            # Process best parameters
            best_params = process_optimization_params(param_keys, best_position)

            score_label = "Validation Total Loss" if is_physics_informed else "Validation MSE"
            print(f"\nBest: {format_params_concise(best_params)} | {score_label}: {best_score:.6g}")
            print("="*80)  # Add clear separator after MoSOA completion

            if not model_specific_results: 
                print(f"WARNING: No successful runs for {model_name} - skipping to next model.")
                continue

            best_run_df = pd.DataFrame(model_specific_results)
            if 'total_loss' not in best_run_df.columns or best_run_df['total_loss'].notna().sum() == 0:
                print(f"WARNING: All optimization runs for {model_name} failed - skipping to next model.")
                continue

            # Get the best run and add MoSOA results
            best_run = best_run_df.loc[best_run_df['total_loss'].idxmin()].to_dict()
            
            # Extract test_score and val_score from metrics dictionaries
            test_score = None
            val_score = None
            if 'test_metrics' in best_run and best_run['test_metrics'] is not None:
                test_score = best_run['test_metrics'].get('mse', None)
            elif 'mse' in best_run:
                test_score = best_run['mse']
            
            if 'val_metrics' in best_run and best_run['val_metrics'] is not None:
                val_score = best_run['val_metrics'].get('mse', None)
            elif 'training_mse' in best_run:
                val_score = best_run['training_mse']
            
            best_run.update({
                'test_score': test_score,
                'val_score': val_score,
                'convergence_history': history,
                'mosoa_best_score': best_score,
                'mosoa_best_params': best_params,
                'iteration_details': iteration_details
            })

            # Create best config from the best parameters
            best_config = copy.deepcopy(base_config)
            for key, value in best_params.items():
                setattr(best_config, key.upper(), value)
            best_config.NUM_BUSES = num_buses

            # Create model kwargs for best model (always OPF mode)
            model_kwargs_best = create_model_kwargs(
                model_config, best_params, num_buses, is_sequential, uses_adaptive_graph, 
                model_name=model_name, config=best_config, normalizer=_normalizer,
                is_physics_informed=is_physics_informed
            )

            # Create data loaders for best model
            loaders_best = create_data_loaders(
                _file_metadata, _adjacency, _ybus_metadata, _normalizer, best_config, 
                is_static=(not is_sequential), topology_cache=_topology_cache, topology_ids=_topology_ids
            )
            _, _, test_loader_best = loaders_best

            # Use the stored model state from the best run (if available)
            try:
                model_to_eval = model_class_map[model_name](**model_kwargs_best).to(device)
                if best_run['model_state'] is not None:
                    model_to_eval.load_state_dict(best_run['model_state'])
                else:
                    print(f"  No model state available for {model_name}, using untrained model for evaluation")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"CUDA OOM during final model creation: {e}")
                    clear_gpu_memory()
                    # Try with even smaller batch size for evaluation
                    best_config.BATCH_SIZE = max(1, best_config.BATCH_SIZE // 2)
                    print(f"Retrying final evaluation with batch size: {best_config.BATCH_SIZE}")
                    # Recreate loaders with smaller batch size
                    loaders_best = create_data_loaders(
                        _file_metadata, _adjacency, _ybus_metadata, _normalizer, best_config, 
                        is_static=(not is_sequential), topology_cache=_topology_cache, topology_ids=_topology_ids
                    )
                    _, _, test_loader_best = loaders_best
                    model_to_eval = model_class_map[model_name](**model_kwargs_best).to(device)
                    model_to_eval.load_state_dict(best_run['model_state'])
                else:
                    raise e

            # Define model output directory (needed for saving results)
            case_name = f"case{num_buses}"
            model_output_dir = os.path.join(
                base_config.CURRENT_RUN_DIR, 
                f"{num_buses}bus", 
                "models",
                model_name
            )
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Save best model checkpoint to model folder
            if best_run['model_state'] is not None:
                best_model_path = os.path.join(model_output_dir, 'best_model.pth')
                torch.save(model_to_eval.state_dict(), best_model_path)
            
            # ==== CONSOLIDATED EVALUATION ====
            # Removed redundant evaluate_model_mc_dropout and compute_engineering_metrics
            # (those metrics are already computed by compute_moopf_metrics)
            
            if is_physics_informed:
                # Evaluate MOOPF objectives (single pass through test data)
                print(f"\n[{model_name}] Running MOOPF evaluation...")
                moopf_results, renewable_impact_data = evaluate_moopf_objectives_normalized(
                    model_to_eval, test_loader_best, best_config, device, _normalizer, is_physics_informed
                )
                
                # Save mse_detailed.csv
                if 'mse_per_sample' in moopf_results:
                    mse_df = pd.DataFrame({'mse_score': moopf_results['mse_per_sample']})
                    mse_detailed_path = os.path.join(model_output_dir, 'mse_detailed.csv')
                    mse_df.to_csv(mse_detailed_path, index=False)
                
                # Save comprehensive results
                # Save comprehensive results
                try:
                    save_results(
                        metrics=moopf_results,
                        results_df=renewable_impact_data,
                        config=base_config,
                        output_dir=model_output_dir
                    )
                except Exception as e:
                    print(f"[{model_name}] Warning: Could not save results: {e}")
                print()  # Add space between MOOPF bar and plot generation bar
            else:
                # Non-physics models: skip MOOPF evaluation
                moopf_results = {}
                renewable_impact_data = None
                print(f"[{model_name}] Non-physics model - skipping MOOPF evaluation")
            
            # Generate all plots with single progress bar (for all models)
            if True: # Always save results
                try:
                    # Get predictions with uncertainty data for visualization
                    _, uncertainty_data = evaluate_model_with_uncertainty(
                        model_to_eval, test_loader_best, device, best_config, _normalizer, is_sequential
                    )
                    
                    # Inject uncertainty into renewable_impact_data for comparative plots
                    if is_physics_informed and renewable_impact_data is not None and not renewable_impact_data.empty:
                        if 'uncertainties' in uncertainty_data and uncertainty_data['uncertainties'] is not None:
                            # Calculate mean uncertainty per sample (scalar)
                            # uncertainties is [n_samples, n_buses, 10]
                            unc_mean = np.mean(uncertainty_data['uncertainties'], axis=(1, 2))
                            if len(unc_mean) == len(renewable_impact_data):
                                renewable_impact_data['uncertainty'] = unc_mean
                            else:
                                print(f"  Warning: Uncertainty length {len(unc_mean)} != Impact length {len(renewable_impact_data)}")
                    
                    # Build list of all plotting tasks
                    plot_tasks = []
                    
                    # Evaluation plots
                    if uncertainty_data.get('renewable_fractions') is not None:
                        plot_tasks.append(('Uncertainty Visualizations', lambda: generate_uncertainty_visualizations(
                            predictions=uncertainty_data['predictions'],
                            targets=uncertainty_data['targets'],
                            renewable_fractions=uncertainty_data['renewable_fractions'],
                            case_name=case_name,
                            output_dir=model_output_dir,
                            model_name=model_name,
                            config=best_config,
                            model_outputs=uncertainty_data.get('model_outputs', None),
                            bus_types=uncertainty_data.get('bus_types', None),
                            timesteps=uncertainty_data.get('timesteps', None)
                        )))
                    
                    if uncertainty_data.get('bus_types') is not None:
                        plot_tasks.append(('Predicted vs Actual', lambda: plot_predicted_vs_actual(
                            predictions=uncertainty_data['predictions'],
                            targets=uncertainty_data['targets'],
                            bus_types=uncertainty_data['bus_types'],
                            case_name=case_name,
                            output_dir=model_output_dir,
                            model_name=model_name
                        )))
                        
                        plot_tasks.append(('Error Distributions', lambda: plot_error_distributions(
                            predictions=uncertainty_data['predictions'],
                            targets=uncertainty_data['targets'],
                            bus_types=uncertainty_data['bus_types'],
                            case_name=case_name,
                            output_dir=model_output_dir,
                            model_name=model_name
                        )))
                        
                        if uncertainty_data['model_outputs'] is not None and uncertainty_data.get('uncertainties') is not None:
                            if uncertainty_data.get('targets_norm') is None:
                                raise ValueError("targets_norm is required for calibration diagram. Cannot plot without normalized targets.")
                            plot_tasks.append(('Calibration Diagram', lambda: plot_calibration_diagram(
                                model_outputs=uncertainty_data['model_outputs'],
                                targets=uncertainty_data['targets'],
                                bus_types=uncertainty_data.get('bus_types', None),
                                case_name=case_name,
                                output_dir=model_output_dir,
                                model_name=model_name,
                                config=best_config,
                                uncertainties=uncertainty_data['uncertainties'],
                                targets_norm=uncertainty_data['targets_norm']
                            )))
                    
                    # Training history plots
                    plot_tasks.append(('Training History', lambda: plot_training_history(
                        history=best_run['training_history'],
                        model_name=model_name,
                        config=best_config,
                        num_buses=num_buses,
                        is_physics_informed=is_physics_informed
                    )))
                    
                    if history:  # history = convergence_curve from MoSOA
                        plot_tasks.append(('MoSOA Convergence', lambda: plot_convergence(
                            history=history,
                            model_name=model_name,
                            config=best_config,
                            num_buses=num_buses
                        )))
                    
                    if is_physics_informed and renewable_impact_data is not None and not renewable_impact_data.empty:
                        plot_tasks.append(('Renewable Impact', lambda: plot_all_renewable_impacts(
                            renewable_impact_data=renewable_impact_data,
                            config=best_config,
                            num_buses=num_buses,
                            model_name=model_name
                        )))
                    
                    # Execute all plotting tasks with single progress bar
                    if plot_tasks:
                        for task_name, task_func in tqdm(plot_tasks, desc=f"Generating plots ({model_name})", unit="plot"):
                            try:
                                task_func()
                            except Exception as e:
                                print(f"  Warning: {task_name} failed: {e}")
                    
                    # Note: Contingency analysis is done during data generation (N-1 scenarios)
                    # The model is already trained on contingency data, no need to re-evaluate
                except Exception as e:
                    print(f"  Warning: Could not generate plots: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate final test performance metric for comparison
            if is_physics_informed and moopf_results:
                # Use MOOPF score for physics-informed models
                final_test_score = moopf_results.get('mse_score', best_run.get('mse', 0.0))
                final_metric_name = "MOOPF Score"
            else:
                # Use test MSE for non-physics models (moopf_results is empty dict)
                final_test_score = best_run.get('mse', best_run.get('test_score', 0.0))
                final_metric_name = "Test MSE"
            
            # Store results for comprehensive summary
            result_entry = {
                'model_name': model_name,
                'num_buses': num_buses,
                'is_physics_informed': is_physics_informed,
                'best_hidden_dim': best_run['HIDDEN_DIM'],
                'best_gc_layers': best_run['NUM_GC_LAYERS'],
                'training_mse': best_run['training_mse'],
                'final_test_score': final_test_score,
                'final_metric_name': final_metric_name,
                'physics_loss': best_run['physics_loss'] if is_physics_informed else 'N/A',
                'safety_loss': best_run['safety_loss'] if is_physics_informed else 'N/A'
            }
            all_results.append(result_entry)
            
            # Track this model for comparative plots
            all_tested_models.append(model_name)
            
            # Print model summary
            print_model_summary(
                best_run, moopf_results, model_name, num_buses, 
                is_physics_informed, final_test_score, final_metric_name
            )

            # Save all results using the training history from the best run
            save_best_model_results(
                best_model=model_to_eval,
                best_run=best_run,
                moopf_results=moopf_results,
                renewable_impact_data=renewable_impact_data,
                training_history=best_run['training_history'],
                config=best_config,
                num_buses=num_buses,
                is_physics_informed=is_physics_informed,
                iteration_details=iteration_details,
                param_keys=param_keys,
                model_name=model_name,
                output_dir=model_output_dir
            )
            
            # Save model_results.csv
            save_model_results_csv(
                best_run=best_run,
                moopf_results=moopf_results,
                config=best_config,
                num_buses=num_buses,
                model_name=model_name,
                output_dir=model_output_dir,
                iteration_details=iteration_details
            )
            
            # Collect data for comparative plots
            if is_physics_informed and not renewable_impact_data.empty:
                bus_renewable_data[model_name] = renewable_impact_data
            
            if history:  # Convergence history
                bus_convergence_data[model_name] = history
            
            clear_gpu_memory()
        
        if True: # Always save results
            print(f"\n Generating plots for {num_buses}-bus...")
            
            # Import comparative visualization functions
            from utils.visualization import create_comparative_renewable_plots, create_comparative_convergence_plot
            
            # Create comparative renewable impact plots for all tested models
            # Always create plots if any models were tested, regardless of physics type
            if all_tested_models:
                try:
                    create_comparative_renewable_plots(bus_renewable_data, base_config, num_buses, all_tested_models)
                except Exception as e:
                    print(f"  Warning: Could not create renewable impact plots: {e}")
            
            # Create comparative convergence plot
            if bus_convergence_data:
                try:
                    create_comparative_convergence_plot(bus_convergence_data, base_config, num_buses)
                except Exception as e:
                    print(f"  Warning: Could not create convergence plots: {e}")
            
            # Copy best model's images to bus system level
            if base_config.DATA_MODE == 'test' and all_results:
                try:
                    # Find best model for this bus system
                    bus_results = [r for r in all_results if r['num_buses'] == num_buses and r['final_test_score'] != float('inf')]
                    if bus_results:
                        best_bus_result = min(bus_results, key=lambda x: x['final_test_score'])
                        best_bus_model_name = best_bus_result['model_name']
                        
                        # Source: model's output directory
                        best_model_dir = os.path.join(
                            base_config.CURRENT_RUN_DIR,
                            f"{num_buses}bus",
                            "models",
                            best_bus_model_name
                        )
                        
                        # Destination: bus system level
                        bus_system_dir = os.path.join(base_config.CURRENT_RUN_DIR, f"{num_buses}bus")
                        
                        # List of all relevant images to copy
                        images_to_copy = [
                            f"{best_bus_model_name}_predicted_vs_actual.png",
                            f"{best_bus_model_name}_error_distributions.png",
                            "calibration_diagram.png",
                            "train_hist.png",
                            "uncertainty_spatial.png",
                            "uncertainty_temporal.png",
                            "mosoa_conv.png"
                        ]
                        
                        # Copy all images if they exist
                        import shutil
                        copied_count = 0
                        for image_file in images_to_copy:
                            src = os.path.join(best_model_dir, image_file)
                            dst = os.path.join(bus_system_dir, image_file)
                            if os.path.exists(src):
                                shutil.copy2(src, dst)
                                copied_count += 1
                        
                        if copied_count > 0:
                            print(f"[Best Model] Copied {copied_count} images from best model ({best_bus_model_name}) to {num_buses}bus folder")
                except Exception as e:
                    print(f"  Warning: Could not copy best model's images: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Final GPU cache clear after completing all models for this bus system
        clear_gpu_memory()
        
        # Clean up data between bus systems
        cleanup_bus_system_data()
    
    # Print comprehensive final summary
    print_comprehensive_summary(all_results, base_config)
    
    # Finalize the run with summary
    if all_results:
        successful_results = [r for r in all_results if r['final_test_score'] != float('inf')]
        
        # Find the actual best model by sorting by final_test_score (lower is better)
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['final_test_score'])
            best_model_name = f"{best_result['model_name']} ({best_result['num_buses']}-bus)"
            best_score_val = best_result['final_test_score']
        else:
            best_model_name = 'None'
            best_score_val = float('inf')
        
        run_summary = {
            'models_tested': [r['model_name'] for r in all_results],
            'total_models': len(all_results),
            'successful_models': len(successful_results),
            'test_config': test_config,
            'best_model': best_model_name,
            'best_score': best_score_val,
            'bus_systems_tested': list(set(r['num_buses'] for r in all_results))
        }
        
        base_config.finalize_run(run_summary)
    else:
        test_config = getattr(base_config, 'test_config', 'all')
        base_config.finalize_run({'status': 'no_results', 'test_config': test_config})


if __name__ == '__main__':
    # Set up signal handlers for clean exit (use flag-based approach)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # === PARALLEL DATA LOADING QUICK START ===
    # To enable parallel data loading, modify the Args class above:
    #
    # For CPU training:
    #   force_cpu = True
    #   parallel_data_loading = True  (recommended)
    #
    # For GPU training on Vast.ai:
    #   force_cpu = False
    #   parallel_data_loading = True  (recommended)
    #
    # Worker count is auto-configured based on your hardware.
    # Set specific number instead of 'auto' for manual control.
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()  # OK at final exit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # OK at final exit
        print("\nTraining script completed")
        close_logger() # Close forensic logger
        sys.exit(0)
