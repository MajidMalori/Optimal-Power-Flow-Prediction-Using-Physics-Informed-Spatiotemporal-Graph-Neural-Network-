import os
import torch
import logging
import numpy as np
import pandas as pd
import copy
import gc
import signal
import sys
import psutil

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
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def log_memory_usage(stage_name):
    """Log memory usage at different stages"""
    # Memory logging disabled
    pass

def get_adaptive_batch_size(num_buses, base_batch_size=32):
    """Get adaptive batch size based on system size"""
    if num_buses >= 118:
        return max(1, base_batch_size // 4)  # Large systems: smaller batches
    elif num_buses >= 57:
        return max(1, base_batch_size // 2)  # Medium systems: medium batches
    else:
        return base_batch_size  # Small systems: full batches

def cleanup_bus_system_data():
    """Clean up data between bus systems to free memory"""
    global _features, _adjacency, _ybus_matrices, _targets
    global _energy_coeffs, _carbon_coeffs, _renewable_fractions
    
    # Clear large data structures
    if '_features' in globals():
        del _features
    if '_adjacency' in globals():
        del _adjacency
    if '_ybus_matrices' in globals():
        del _ybus_matrices
    if '_targets' in globals():
        del _targets
    if '_energy_coeffs' in globals():
        del _energy_coeffs
    if '_carbon_coeffs' in globals():
        del _carbon_coeffs
    if '_renewable_fractions' in globals():
        del _renewable_fractions
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def cleanup_model_resources(model, trainer, optimizer, criterion):
    """Clean up model resources between different models"""
    # Delete model and related objects
    if model is not None:
        del model
    if trainer is not None:
        del trainer
    if optimizer is not None:
        del optimizer
    if criterion is not None:
        del criterion
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for memory efficiency"""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled for memory efficiency")

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
import matplotlib.pyplot as plt

from utils.data_loader import load_power_system_data, create_data_loaders
from utils.metrics import PowerSystemLoss
from utils.data_validation import validate_data_before_training
from utils.optimization import (mosoa_optimizer, trial_based_search, setup_hyperparameter_bounds, create_model_kwargs, 
                               generate_run_name, process_optimization_params, format_params_concise,
                               calculate_objective_score)
from utils.evaluation import (evaluate_model, evaluate_model_normalized, evaluate_model_with_uncertainty,
                             evaluate_moopf_objectives, evaluate_moopf_objectives_normalized, 
                             save_best_model_results, print_comprehensive_summary, print_model_summary)
from utils.uncertainty_analysis import generate_uncertainty_visualizations
from utils.data_profile_story import analyze_data_profiles
from trainers.model_trainer import PowerSystemTrainer
from config import Config, Args


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
    """Handle interrupt signals to ensure proper cleanup."""
    print(f"\nReceived signal {signum}. Cleaning up...")
    if _config_instance:
        _config_instance.finalize_run({'status': 'interrupted', 'reason': f'signal_{signum}'})
    sys.exit(0)

def setup_professional_logging():
    """Setup professional logging with memory tracking"""
    # Use console logging only - detailed logs are already saved in experimental_results/
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
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Set PyTorch CUDA memory allocation configuration to prevent fragmentation
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent memory fragmentation")
    
    # Initialize arguments from centralized config
    args = Args()
    base_config = Config(
        data_mode=args.data_mode, 
        save_results=args.save_results, 
        test_timesteps=args.test_timesteps, 
        clear_results=args.clear_results,
        hours_per_day=args.hours_per_day,
        sequence_length=args.sequence_length
    )
    _config_instance = base_config  # Store for signal handler
    
    # Parse bus systems to test
    def parse_bus_systems(bus_systems_arg):
        """Parse bus systems argument and return list of bus numbers to test."""
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
                        print(f"Warning: {bus_num}-bus system not available. Available: {base_config.NUM_BUSES}")
                except ValueError:
                    print(f"Warning: Invalid bus system '{bus_str}'. Skipping.")
            return bus_list if bus_list else base_config.NUM_BUSES
    
    bus_systems_to_test = parse_bus_systems(args.bus_systems)
    
    # Track all results for comprehensive summary
    all_results = []
    
    # STEP 1: Print run information
    run_info = base_config.get_run_info()
    actual_timesteps = base_config.DATA_MODE_TIMESTEPS[args.data_mode]
    print(f"\nRUN: {run_info['run_id']} | Config: {args.test_config} | Mode: {args.data_mode.upper()} ({actual_timesteps} timesteps)")
    print(f"Bus Systems: {bus_systems_to_test}")
    print("="*80)
    
    # STEP 2: Validate data before training (check files exist, generate if missing, show convergence)
    if not validate_data_before_training(base_config, bus_systems_to_test):
        print("Data validation failed. Exiting training.")
        return
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # === DEVICE AND PARALLEL CONFIGURATION ===
    device, device_reason = get_safe_device(args.force_cpu, min_free_memory_gb=2.0)
    is_gpu = device.type == 'cuda'
    
    # Configure hardware
    clear_gpu_memory()
    log_memory_usage("Initial startup")
    
    def get_optimal_workers():
        if is_gpu and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return gpu_memory, torch.cuda.get_device_name(0), 8 if gpu_memory >= 12 else (6 if gpu_memory >= 8 else 4)
        else:
            import psutil
            cpu_count = psutil.cpu_count(logical=True)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            # For small systems (2-3 cores), use all cores. For larger, leave one for system.
            if cpu_count >= 8:
                workers = 4  # Cap at 4 for very large systems
            elif cpu_count >= 4:
                workers = cpu_count - 1  # Leave one for system
            elif cpu_count >= 2:
                workers = cpu_count  # Use all cores for 2-3 core systems
            else:
                workers = 1  # Single core fallback
            return memory_gb, f"CPU {cpu_count}c", workers
    
    hw_size, hw_name, data_workers = get_optimal_workers()
    data_workers = data_workers if args.data_workers == 'auto' else args.data_workers
    base_config.NUM_WORKERS = data_workers if args.parallel_data_loading else 0
    
    print(f"{hw_name} ({hw_size:.1f} GB) | {base_config.NUM_WORKERS} workers")
    print("="*80)

    # Get model configurations from config
    model_class_map = base_config.get_model_class_map()
    model_config_map = base_config.model_config_map
    models_to_test = base_config.get_models_to_test(args.test_config)
    
    # Filter models based on user selection
    if args.models_to_train != 'all':
        selected_models = [m.strip() for m in args.models_to_train.split(',')]
        models_to_test = [m for m in models_to_test if m in selected_models]
        print(f"Selected models to train: {models_to_test}")
        if not models_to_test:
            print("ERROR: No valid models selected. Available models: PIGCLSTM, PIGCGRU, ResnetPIGCLSTM, ResnetPIGCGRU")
            return

    # === MAIN TRAINING EXECUTION ===
    print(f"\n TRAINING: {len(bus_systems_to_test)} bus systems {bus_systems_to_test}\n")
    
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
        # This ensures correct per-unit calculations: Case33=10MVA, Case57/118=100MVA
        base_config.CASE_NAME = case_name
        try:
            data_tuple = load_power_system_data(base_config, case_name)
            _features, _adjacency, _ybus_matrices, _targets, _bus_types, _energy_coeffs, _carbon_coeffs, _renewable_fractions, _normalizer = data_tuple
            
            # Generate data profile story if enabled
            if base_config.GENERATE_DATA_PROFILE_STORY:
                try:
                    analyze_data_profiles(
                        config=base_config,
                        case_name=case_name,
                        features=_features,  # Use current data mode (train/test)
                        normalizer=_normalizer,
                        renewable_fractions=_renewable_fractions
                    )
                except Exception as e:
                    print(f"  Warning: Could not generate data profile story: {e}")
                    import traceback
                    traceback.print_exc()
            
            bus_models_to_test = models_to_test.copy()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        for model_name in bus_models_to_test:
            print(f"\n{'='*60}")
            print(f"{model_name} on {num_buses}-bus")
            print(f"{'='*60}")
            
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

            def objective_function(params_array):
                params = process_optimization_params(param_keys, params_array)

                run_config = copy.deepcopy(base_config)
                for key, value in params.items(): 
                    setattr(run_config, key.upper(), value)
                run_config.NUM_BUSES = num_buses

                run_name = generate_run_name(model_name, params, num_buses, is_sequential)

                try:
                    if run_config.SAVE_RESULTS:
                        setup_logging(run_config.get_evaluation_path(f"{num_buses}bus/logs/{run_name}.log"))
                    
                    # Clear GPU memory before starting
                    clear_gpu_memory()
                    
                    # Check memory before model creation (monitoring disabled)
                    
                    # Use adaptive batch size based on system size
                    run_config.BATCH_SIZE = get_adaptive_batch_size(num_buses, run_config.BATCH_SIZE)
                    log_memory_usage(f"Before loading {num_buses}-bus data")
                    
                    loaders = create_data_loaders(
                        _features, _adjacency, _ybus_matrices, _targets,
                        _energy_coeffs, _carbon_coeffs, _renewable_fractions, base_config, is_static=(not is_sequential),
                        bus_types=_bus_types
                    )
                    train_loader, val_loader, test_loader = loaders

                    # Create model with optimized parameters
                    # Detect OPF mode: check if bus_types exist (OPF) vs None (state estimation)
                    is_opf_mode = (_bus_types is not None)
                    model_kwargs = create_model_kwargs(
                        model_config, params, num_buses, is_sequential, uses_adaptive_graph, 
                        model_name=model_name, is_opf_mode=is_opf_mode
                    )
                    
                    # Check memory before model creation
                    if torch.cuda.is_available():
                        memory_info = check_gpu_memory()
                        if memory_info['free'] < 1024**3:  # Less than 1GB free
                            print("Warning: Low GPU memory, clearing cache...")
                            clear_gpu_memory()
                    
                    # Create model with error handling for OOM
                    log_memory_usage(f"Before creating {model_name} model")
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
                                _features, _adjacency, _ybus_matrices, _targets, 
                                _energy_coeffs, _carbon_coeffs, _renewable_fractions, run_config, 
                                is_static=(not is_sequential), bus_types=_bus_types
                            )
                            train_loader, val_loader, test_loader = loaders
                            model = model_class_map[model_name](**model_kwargs).to(device)
                        else:
                            raise e
                    
                    # Use appropriate loss function based on whether model is physics-informed
                    criterion = PowerSystemLoss(
                        config=run_config, 
                        normalizer=_normalizer, 
                        is_gcn=(not is_physics_informed)
                    ).to(device)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.LEARNING_RATE)

                    trainer = PowerSystemTrainer(model, criterion, optimizer, run_config, device, is_physics_informed)
                    
                    # Train the model
                    trainer.train(train_loader, val_loader)

                    # Get validation metrics for hyperparameter optimization (use normalized data like training)
                    val_metrics = evaluate_model_normalized(model, val_loader, device, run_config, _normalizer, is_sequential)
                    
                    # Get test metrics for final evaluation
                    test_metrics = evaluate_model(model, test_loader, device, run_config, _normalizer, is_sequential)

                    # Calculate total loss for optimization using validation metrics
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
                    
                    run_results = {
                        'run_name': run_name, 
                        'model_name': model_name, 
                        **params, 
                        **test_metrics,  # Final test performance for reporting
                        'val_metrics': val_metrics,  # Validation metrics used for optimization
                        'total_loss': total_loss,  # Based on validation metrics
                        'training_mse': val_metrics['mse'],
                        'training_history': trainer.get_training_history(),
                        'model_state': model_state,  # May be None for large models
                        'model_config': run_config  
                    }
                    model_specific_results.append(run_results)

                    return total_loss
                    
                except Exception as e:
                    logging.error(f"Run {run_name} failed: {e}", exc_info=True)
                    return float('inf')
                finally:
                    # Clean up model and memory
                    cleanup_model_resources(
                        locals().get('model'), 
                        locals().get('trainer'), 
                        locals().get('optimizer'), 
                        locals().get('criterion')
                    )
                    log_memory_usage(f"After {model_name} training")

            if args.use_mosoa:
                print(f"Optimizing with MoSOA: {mosoa_params['num_seagulls']} seagulls × {mosoa_params['max_iterations']} iterations")
                best_score, best_position, history, iteration_details = mosoa_optimizer(
                    mosoa_params['num_seagulls'], 
                    mosoa_params['max_iterations'], 
                    lower_bounds, upper_bounds, dim, objective_function,
                    param_keys=param_keys
                )
            else:
                print(f"Optimizing with trial-based search: {args.num_trials} trials")
                best_score, best_position, history, iteration_details = trial_based_search(
                    num_trials=args.num_trials,
                    lower_bound=lower_bounds,
                    upper_bound=upper_bounds,
                    dim=dim,
                    objective_func=objective_function,
                    search_strategy='latin_hypercube'
                )

            # Process best parameters
            best_params = process_optimization_params(param_keys, best_position)

            print(f"\nBest: {format_params_concise(best_params)} | Score: {best_score:.6g}")
            print("="*80)  # Add clear separator after MoSOA completion

            if not model_specific_results: 
                print(f"No successful runs for {model_name}.")
                continue

            best_run_df = pd.DataFrame(model_specific_results)
            if 'total_loss' not in best_run_df.columns or best_run_df['total_loss'].notna().sum() == 0:
                print(f"All runs for {model_name} failed.")
                continue

            # Get the best run and add MoSOA results
            best_run = best_run_df.loc[best_run_df['total_loss'].idxmin()].to_dict()
            best_run.update({
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

            # Create model kwargs for best model
            # Detect OPF mode: check if bus_types exist (OPF) vs None (state estimation)
            is_opf_mode = (_bus_types is not None)
            model_kwargs_best = create_model_kwargs(
                model_config, best_params, num_buses, is_sequential, uses_adaptive_graph, 
                model_name=model_name, is_opf_mode=is_opf_mode
            )

            # Create data loaders for best model
            loaders_best = create_data_loaders(
                _features, _adjacency, _ybus_matrices, _targets, 
                _energy_coeffs, _carbon_coeffs, _renewable_fractions, best_config, 
                is_static=(not is_sequential), bus_types=_bus_types
            )
            _, _, test_loader_best = loaders_best

            # Use the stored model state from the best run (if available)
            try:
                model_to_eval = model_class_map[model_name](**model_kwargs_best).to(device)
                if best_run.get('model_state') is not None:
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
                        _features, _adjacency, _ybus_matrices, _targets, 
                        _energy_coeffs, _carbon_coeffs, _renewable_fractions, best_config, 
                        is_static=(not is_sequential), bus_types=_bus_types
                    )
                    _, _, test_loader_best = loaders_best
                    model_to_eval = model_class_map[model_name](**model_kwargs_best).to(device)
                    model_to_eval.load_state_dict(best_run['model_state'])
                else:
                    raise e

            # Evaluate MOOPF objectives for the best model (using normalized data for consistent scoring)
            moopf_results, renewable_impact_data = evaluate_moopf_objectives_normalized(
                model_to_eval, test_loader_best, best_config, device, _normalizer, is_physics_informed
            )
            
            # Generate uncertainty visualizations
            if base_config.SAVE_RESULTS and base_config.DATA_MODE == 'test':
                try:
                    # Get predictions with uncertainty data (silent generation)
                    _, uncertainty_data = evaluate_model_with_uncertainty(
                        model_to_eval, test_loader_best, device, best_config, _normalizer, is_sequential
                    )
                    
                    # Generate and save uncertainty graphs in model-specific folder
                    case_name = f"case{num_buses}"
                    model_output_dir = os.path.join(
                        base_config.CURRENT_RUN_DIR, 
                        f"{num_buses}bus", 
                        "models", 
                        model_name
                    )
                    os.makedirs(model_output_dir, exist_ok=True)
                    
                    generate_uncertainty_visualizations(
                        predictions=uncertainty_data['predictions'],
                        targets=uncertainty_data['targets'],
                        renewable_fractions=uncertainty_data['renewable_fractions'],
                        case_name=case_name,
                        output_dir=model_output_dir,
                        model_name=model_name,
                        config=best_config,  # Pass config for time-series mode detection
                        bus_types=uncertainty_data.get('bus_types', None)  # Pass bus_types for OPF mode
                    )
                except Exception as e:
                    print(f"  Warning: Could not generate uncertainty visualizations: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Calculate final test performance metric for comparison
            if is_physics_informed:
                # Use MSE as the primary metric, but track MOOPF score separately
                final_test_score = moopf_results['mse_score'].mean() if 'mse_score' in moopf_results.columns else best_run.get('mse', float('inf'))
                final_metric_name = "MOOPF Score"  # This is just a label indicating we also have MOOPF evaluation
            else:
                final_test_score = moopf_results['mse_score'].mean()
                final_metric_name = "Test MSE"
            
            # Store results for comprehensive summary
            result_entry = {
                'model_name': model_name,
                'num_buses': num_buses,
                'is_physics_informed': is_physics_informed,
                'best_hidden_dim': best_run.get('HIDDEN_DIM', 'N/A'),
                'best_gc_layers': best_run.get('NUM_GC_LAYERS', 'N/A'),
                'training_mse': best_run.get('training_mse', best_run.get('mse', float('inf'))),
                'final_test_score': final_test_score,
                'final_metric_name': final_metric_name,
                'power_violation': best_run.get('power_violation', 'N/A') if is_physics_informed else 'N/A',
                'voltage_violation': best_run.get('voltage_violation', 'N/A') if is_physics_informed else 'N/A'
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
                param_keys=param_keys
            )
            
            # Collect data for comparative plots
            if is_physics_informed and not renewable_impact_data.empty:
                bus_renewable_data[model_name] = renewable_impact_data
            
            if history:  # Convergence history
                bus_convergence_data[model_name] = history
            
            clear_gpu_memory()
        
        if base_config.SAVE_RESULTS:
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
            
            # Copy best model's uncertainty graphs to bus system level
            if base_config.DATA_MODE == 'test' and all_results:
                try:
                    # Find best model for this bus system
                    bus_results = [r for r in all_results if r['num_buses'] == num_buses and r['final_test_score'] != float('inf')]
                    if bus_results:
                        best_bus_result = min(bus_results, key=lambda x: x['final_test_score'])
                        best_bus_model_name = best_bus_result['model_name']
                        
                        # Source: model's uncertainty graphs
                        model_uncertainty_dir = os.path.join(
                            base_config.CURRENT_RUN_DIR,
                            f"{num_buses}bus",
                            "models",
                            best_bus_model_name
                        )
                        
                        # Destination: bus system level
                        bus_system_dir = os.path.join(base_config.CURRENT_RUN_DIR, f"{num_buses}bus")
                        
                        # Copy uncertainty graphs if they exist
                        import shutil
                        copied_count = 0
                        for uncertainty_file in ['uncertainty_spatial.png', 'uncertainty_temporal.png']:
                            src = os.path.join(model_uncertainty_dir, uncertainty_file)
                            dst = os.path.join(bus_system_dir, uncertainty_file)
                            if os.path.exists(src):
                                shutil.copy2(src, dst)
                                copied_count += 1
                        if copied_count > 0:
                            print(f"[Uncertainty] Copied {copied_count} plots from best model ({best_bus_model_name}) to {num_buses}bus folder")
                except Exception as e:
                    print(f"  Warning: Could not copy best model's uncertainty graphs: {e}")
        
        # Final GPU cache clear after completing all models for this bus system
        clear_gpu_memory()
        log_memory_usage(f"After completing {num_buses}-bus system")
        
        # Clean up data between bus systems
        cleanup_bus_system_data()
        log_memory_usage("After bus system cleanup")
    
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
            'test_config': args.test_config,
            'best_model': best_model_name,
            'best_score': best_score_val,
            'bus_systems_tested': list(set(r['num_buses'] for r in all_results))
        }
        
        base_config.finalize_run(run_summary)
    else:
        base_config.finalize_run({'status': 'no_results', 'test_config': args.test_config})


def signal_handler(signum, _):
    """Handle interrupt signals gracefully"""
    print(f"\n Received signal {signum}, cleaning up...")
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

if __name__ == '__main__':
    # Set up signal handlers for clean exit
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
        # Ensure clean exit
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nTraining script completed")
        sys.exit(0)
