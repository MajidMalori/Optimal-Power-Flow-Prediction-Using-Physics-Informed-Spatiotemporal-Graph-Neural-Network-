import os
import torch
import logging
import numpy as np
import pandas as pd
import copy
import sys
import yaml
import signal
import gc
import random
import shutil
from tqdm import tqdm

# Set matplotlib backend before any plotting imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent threading issues

# Project-specific imports
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
from utils.shutdown_flag import set_shutdown, get_shutdown
from config import Config

# Global data cache for bus systems (used across multiple functions)
_file_metadata = _adjacency = _ybus_metadata = _normalizer = _topology_cache = _topology_ids = None




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






def get_device():
    """Get the best available device (GPU if available, otherwise CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda'), 'gpu_available'
    return torch.device('cpu'), 'no_cuda'


# Global variable to store config for signal handler
_config_instance = None

def signal_handler(signum, _):
    """Handle interrupt signals gracefully - set flag instead of printing directly."""
    set_shutdown()

def setup_logging():
    """Setup logging with memory tracking"""
    
    # Custom StreamHandler that uses tqdm.write() to avoid breaking progress bars
    class TqdmLoggingHandler(logging.StreamHandler):
        """Logging handler that uses tqdm.write() to avoid breaking progress bars"""
        def emit(self, record):
            try:
                msg = self.format(record)
                # Filter out noisy pandapower messages that break progress bars
                if "dtypes could not be corrected" in msg:
                    return  # Suppress this specific noisy message
                # Use tqdm.write() to properly handle output during progress bar display
                tqdm.write(msg, file=sys.stderr)
                self.flush()
            except Exception:
                self.handleError(record)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[TqdmLoggingHandler()]
    )
    
    # Suppress noisy pandapower INFO messages that interfere with progress bars
    logging.getLogger('pandapower').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing for the model to save memory.
    Trades compute for memory: saves memory by re-computing activations during backward pass.
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    elif hasattr(model, 'set_gradient_checkpointing'):
        model.set_gradient_checkpointing(True)
    else:
        # Generic approach for modules that support it
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True

def main():
    global _config_instance
    
    # Setup logging and signal handlers
    logger = setup_logging()
    logger.info("Starting training session")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Environment setup
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Config setup
    cli_args = Config.parse_cli_args()
    base_config = Config(cli_args=cli_args, data_mode='train', load_yaml=True)
    base_config.create_run_directories()
    _config_instance = base_config
    
    # Parse bus systems to test (from Config, not Args)
    def parse_bus_systems(bus_systems_arg):
        """Parse bus systems argument."""
        if isinstance(bus_systems_arg, list):
            return [int(b) for b in bus_systems_arg]
        if isinstance(bus_systems_arg, int):
            return [bus_systems_arg]
        if isinstance(bus_systems_arg, str):
            if bus_systems_arg.lower() == 'all':
                return base_config.NUM_BUSES
            try:
                bus_list = [int(b.strip()) for b in bus_systems_arg.split(',') if int(b.strip()) in base_config.NUM_BUSES]
                if not bus_list: print(f"WARNING: No valid bus systems found in '{bus_systems_arg}'. Defaulting to all.")
                return bus_list if bus_list else base_config.NUM_BUSES
            except ValueError:
                print(f"WARNING: Invalid bus system string '{bus_systems_arg}'. Defaulting to all.")
        return base_config.NUM_BUSES
    
    bus_systems_to_test = parse_bus_systems(getattr(Config, 'bus_systems', 'all'))
    all_results = []
    
    if not validate_data_before_training(base_config, bus_systems_to_test):
        raise RuntimeError("Data validation failed. Run data/main.py first.")
    
    # Reproducibility
    seed = getattr(base_config, 'SEED', 42)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print(f"\n[Seed] {seed}")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError: pass
    
    # Device setup
    device, _ = get_device()
    clear_gpu_memory()
    
    base_config.NUM_WORKERS = int(getattr(base_config, 'NUM_WORKERS', 0)) if getattr(base_config, 'EXPERIMENTAL_PARALLEL_DATA_LOADING', True) else 0
    
    device_info = f"CUDA ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB)" if device.type == 'cuda' else "CPU"
    print(f"RUN: {base_config.get_run_info()['run_id']} | Config: {getattr(Config, 'test_config', 'all')} | Mode: {getattr(Config, 'data_mode', 'test').upper()} | Buses: {bus_systems_to_test} | Device: {device_info} | Workers: {base_config.NUM_WORKERS}")

    # Model selection
    model_class_map = base_config.get_model_class_map()
    model_config_map = base_config.model_config_map
    models_to_train = getattr(Config, 'models_to_train', None)
    
    if models_to_train and models_to_train != "all":
        models_to_test = [m.strip() for m in models_to_train.split(',')] if isinstance(models_to_train, str) else (models_to_train if isinstance(models_to_train, list) else [models_to_train])
        print(f"[Models] Explicit: {models_to_test}")
    else:
        models_to_test = base_config.get_models_to_test(getattr(Config, 'test_config', 'all'))
        print(f"[Models] Config: {models_to_test}")
    
    if not models_to_test: raise ValueError("No models selected for training.")

    # === MAIN TRAINING EXECUTION ===
    for num_buses in bus_systems_to_test:
        # Check for shutdown flag before starting each bus system
        if get_shutdown():
            print("\nShutdown signal received - exiting training")
            raise KeyboardInterrupt("Training interrupted by user")
        
        # Get adaptive MoSOA parameters for this system size
        mosoa_params = base_config._ModelConfig.get_adaptive_mosoa_params(num_buses)
        
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
            print(f"WARNING: Skipping {num_buses}-bus - data not found. Run: python data/main.py")
            continue

        for model_name in bus_models_to_test:
            # Check for shutdown flag before starting each model
            if get_shutdown():
                print("\nShutdown signal received - exiting training")
                raise KeyboardInterrupt("Training interrupted by user")
            
            print(f"\n\n{'='*80}")
            print(f"  {model_name} - {num_buses}-bus system")
            print(f"{'='*80}")
            
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
                # Check for shutdown flag before starting each training run
                if get_shutdown():
                    print("\nShutdown signal received - exiting training")
                    raise KeyboardInterrupt("Training interrupted by user")
                
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
                    
                    # Create model
                    try:
                        model = model_class_map[model_name](**model_kwargs).to(device)
                        if getattr(run_config, 'GRADIENT_CHECKPOINTING', False): enable_gradient_checkpointing(model)
                    except RuntimeError as e:
                        if "out of memory" not in str(e).lower(): raise e
                        print(f"CUDA OOM: {e}. Retrying with batch size {run_config.BATCH_SIZE // 2}")
                        clear_gpu_memory()
                        run_config.BATCH_SIZE = max(1, run_config.BATCH_SIZE // 2)
                        loaders = create_data_loaders(_file_metadata, _adjacency, _ybus_metadata, _normalizer, run_config, is_static=(not is_sequential), topology_cache=_topology_cache, topology_ids=_topology_ids)
                        train_loader, val_loader, test_loader = loaders
                        model = model_class_map[model_name](**model_kwargs).to(device)

                    criterion = PowerSystemLoss(config=run_config, normalizer=_normalizer, is_gcn=(not is_physics_informed)).to(device)
                    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=run_config.LEARNING_RATE, weight_decay=getattr(run_config, 'WEIGHT_DECAY', 0.0001))
                    trainer = PowerSystemTrainer(model, criterion, optimizer, run_config, device, is_physics_informed)
                    
                    # Config logging
                    config_params = {
                        'lr': run_config.LEARNING_RATE, 'batch': run_config.BATCH_SIZE, 
                        'accum': getattr(run_config, 'GRADIENT_ACCUMULATION_STEPS', 1), 'epochs': run_config.NUM_EPOCHS,
                        'patience': getattr(run_config, 'EARLY_STOPPING_PATIENCE', 10), **params
                    }
                    config_str = ", ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(config_params.items())])
                    iter_info = f" [Iter {mosoa_iter[0]}/{mosoa_max_iter} | Run {((mosoa_run_total[0] - 1) % mosoa_runs_per_iter) + 1}/{mosoa_runs_per_iter}]"
                    print(f"  Config: {config_str}{iter_info}\n")
                    
                    trainer.train(train_loader, val_loader, model_name=model_name, num_buses=num_buses, config_params=config_params)
                    print()

                    training_history = trainer.get_training_history()
                    
                    if is_physics_informed and 'val_total_loss' in training_history:
                        val_metrics = {'total_loss': float(training_history['val_total_loss'][-1]), 'mse': float(training_history['val_mse'][-1])}
                    else:
                        val_metrics = evaluate_performance(model, val_loader, device, run_config, _normalizer, is_sequential, return_denormalized=False)
                    
                    test_metrics = evaluate_performance(model, test_loader, device, run_config, _normalizer, is_sequential, return_denormalized=False)
                    total_loss = calculate_objective_score(val_metrics, run_config, is_physics_informed)

                    # Store results
                    model_state = None
                    try:
                        state_size = sum(p.numel() * p.element_size() for p in model.parameters())
                        if state_size <= 100 * 1024 * 1024: model_state = model.state_dict()
                        else: print(f"  Model state too large ({state_size / 1024**2:.1f} MB), skipping save")
                    except Exception as e: print(f"  Could not save model state: {e}")
                    
                    run_results = {
                        'run_name': run_name, 'model_name': model_name, **params, **test_metrics,
                        'val_metrics': val_metrics, 'total_loss': total_loss, 'training_mse': val_metrics['mse'],
                        'physics_loss': training_history['val_physics_loss'][-1], 'safety_loss': training_history['val_safety_loss'][-1],
                        'training_history': training_history, 'model_state': model_state, 'model_config': run_config  
                    }
                    model_specific_results.append(run_results)
                    return total_loss
                    
                except KeyboardInterrupt:
                    # Re-raise KeyboardInterrupt to allow proper shutdown
                    raise
                except Exception as e:
                    # Sanitize error message for Windows encoding compatibility
                    error_msg = str(e).replace('η', 'eta').replace('δ', 'delta').replace('σ', 'sigma').replace('λ', 'lambda')
                    logging.error(f"Run {run_name} failed: {error_msg}", exc_info=True)
                    return float('inf')
                finally:
                    # Delete DataLoaders to kill worker processes immediately
                    if 'loaders' in locals(): del loaders
                    if 'train_loader' in locals(): del train_loader
                    if 'val_loader' in locals(): del val_loader
                    if 'test_loader' in locals(): del test_loader
                    
                    # Delete model and trainer to free GPU memory
                    if 'model' in locals(): del model
                    if 'trainer' in locals(): del trainer
                    if 'optimizer' in locals(): del optimizer
                    if 'criterion' in locals(): del criterion
                    
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Always use MoSOA for hyperparameter optimization
            print(f"\nMoSOA: {mosoa_params['num_seagulls']} seagulls × {mosoa_params['max_iterations']} iterations ({mosoa_params['strategy']})\n")
            
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
            
            # Check for shutdown flag before starting MoSOA
            if get_shutdown():
                print("\nShutdown signal received - exiting training")
                raise KeyboardInterrupt("Training interrupted by user")
            
            best_score, best_position, history, iteration_details = mosoa_optimizer(
                mosoa_params['num_seagulls'], 
                mosoa_params['max_iterations'], 
                lower_bounds, upper_bounds, dim, objective_with_tracking,
                param_keys=param_keys
            )


            # Process best parameters
            best_params = process_optimization_params(param_keys, best_position)

            score_label = "Val Loss" if is_physics_informed else "Val MSE"
            print(f"Best: {format_params_concise(best_params)} | {score_label}: {best_score:.6g}")

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
            # Evaluate MOOPF objectives (single pass through test data)
            print(f"[{model_name}] MOOPF evaluation...")
            moopf_results, renewable_impact_data = evaluate_moopf_objectives_normalized(
                model_to_eval, test_loader_best, best_config, device, _normalizer, is_physics_informed=True # Force True to enable evaluation
            )
            
            # Save mse_detailed.csv
            if 'mse_per_sample' in moopf_results:
                mse_df = pd.DataFrame({'mse_score': moopf_results['mse_per_sample']})
                mse_detailed_path = os.path.join(model_output_dir, 'mse_detailed.csv')
                mse_df.to_csv(mse_detailed_path, index=False)
            
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
            
            # Generate all plots with single progress bar (for all models)
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
                # Configure tqdm to handle logging output properly
                import sys
                if plot_tasks:
                    # Use file=sys.stdout (default) for progress bar
                    # The TqdmLoggingHandler will route logging to stderr via tqdm.write()
                    # This keeps progress bar on stdout and logging on stderr (proper separation)
                    plot_pbar = tqdm(
                        plot_tasks, 
                        desc=f"Generating plots ({model_name})", 
                        unit="plot",
                        file=sys.stdout,  # Progress bar on stdout
                        dynamic_ncols=True,
                        mininterval=0.1,  # Update frequently for responsive display
                        leave=True,  # Keep bar when done for visibility
                        ncols=None,  # Auto-detect terminal width
                        disable=False  # Ensure it's enabled
                    )
                    # Force immediate display
                    plot_pbar.refresh()
                    
                    for task_name, task_func in plot_pbar:
                        try:
                            # Update description to show current task
                            plot_pbar.set_description(f"Generating plots ({model_name}): {task_name}")
                            plot_pbar.refresh()  # Force immediate update before task
                            task_func()
                            # Refresh after task completes
                            plot_pbar.refresh()
                        except Exception as e:
                            # Use tqdm.write() to avoid breaking progress bar
                            tqdm.write(f"  Warning: {task_name} failed: {e}", file=sys.stderr)
                            plot_pbar.refresh()  # Refresh after error message
                    
                    # Final update
                    plot_pbar.set_description(f"Generating plots ({model_name}): Complete")
                    plot_pbar.refresh()
                    plot_pbar.close()
                
                # Note: Contingency analysis is done during data generation (N-1 scenarios)
                # The model is already trained on contingency data, no need to re-evaluate
            except Exception as e:
                print(f"  Warning: Could not generate plots: {e}")
                import traceback
                traceback.print_exc()
            
            # Calculate final test performance metric for comparison
            # Use MOOPF Score for comparison to ensure fair ranking
            if moopf_results:
                final_test_score = moopf_results.get('moopf_score', best_run.get('mse', 0.0))
                final_metric_name = "MOOPF Score"
            else:
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
                'safety_loss': best_run['safety_loss'] if is_physics_informed else 'N/A',
                # Add detailed MOOPF metrics
                'test_mse': best_run.get('test_score', 'N/A'),
                'moopf_score': moopf_results.get('moopf_score', 'N/A'),
                'power_loss': moopf_results.get('power_loss', 'N/A'),
                'voltage_deviation': moopf_results.get('voltage_deviation', 'N/A'),
                'carbon_emissions': moopf_results.get('carbon_emissions', 'N/A')
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
        
        # Import comparative visualization functions
        from utils.visualization import (
            create_comparative_renewable_plots, 
            create_comparative_convergence_plot,
            create_moopf_comparison_plot
        )
        
        # Create comparative renewable impact plots for all tested models
        # Always create plots if any models were tested, regardless of physics type
        if all_tested_models:
            try:
                create_comparative_renewable_plots(bus_renewable_data, base_config, num_buses, all_tested_models)
            except Exception as e:
                print(f"  Warning: Could not create renewable impact plots: {e}")
                
            try:
                # Create MOOPF metrics comparison plot (Power Loss, Voltage Deviation, Carbon)
                # This plots for ALL models (Physics and Non-Physics) using data from all_results
                bus_output_dir = os.path.join(base_config.CURRENT_RUN_DIR, f"{num_buses}bus")
                create_moopf_comparison_plot(all_results, bus_output_dir, num_buses)
            except Exception as e:
                print(f"  Warning: Could not create MOOPF comparison plot: {e}")
        
        # Create comparative convergence plot
        if bus_convergence_data:
            try:
                create_comparative_convergence_plot(bus_convergence_data, base_config, num_buses)
            except Exception as e:
                print(f"  Warning: Could not create convergence plots: {e}")
        
        # Copy best model's images to bus system level (for both train and test modes)
        if all_results:
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
                        print(f"\n[Best Model] Copied {copied_count} images from best model ({best_bus_model_name}) to {num_buses}bus folder")
            except Exception as e:
                print(f"  Warning: Could not copy best model's images: {e}")
                import traceback
                traceback.print_exc()
        
        # Final GPU cache clear after completing all models for this bus system
        clear_gpu_memory()
        
        # Clean up data between bus systems
        cleanup_bus_system_data()
    
    # Final summary
    if all_results:
        successful = [r for r in all_results if r['final_test_score'] != float('inf')]
        best = min(successful, key=lambda x: x['final_test_score']) if successful else None
        base_config.finalize_run({
            'models_tested': [r['model_name'] for r in all_results], 'total': len(all_results), 'success': len(successful),
            'test_config': test_config, 'best_model': f"{best['model_name']} ({best['num_buses']}b)" if best else 'None',
            'best_score': best['final_test_score'] if best else float('inf'), 'buses': list(set(r['num_buses'] for r in all_results))
        })
    else:
        base_config.finalize_run({'status': 'no_results', 'test_config': getattr(base_config, 'test_config', 'all')})

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler); signal.signal(signal.SIGTERM, signal_handler)
    try: main()
    except KeyboardInterrupt: print("\nTraining interrupted by user")
    except Exception as e: print(f"\nTraining failed with error: {e}"); import traceback; traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("\nTraining script completed\n"); sys.exit(0)
