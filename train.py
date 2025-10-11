import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import gc
import time
import signal
import sys
# Removed ThreadPoolExecutor imports - parallel bus systems disabled

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
    
    print(f"🎮 GPU Memory: {free_memory_gb:.2f} GB free / {total_memory_gb:.2f} GB total")
    
    # If free memory is less than minimum required, fallback to CPU
    if free_memory_gb < min_free_memory_gb:
        print(f"⚠️  GPU memory insufficient ({free_memory_gb:.2f} GB < {min_free_memory_gb} GB), falling back to CPU")
        return torch.device('cpu'), 'insufficient_gpu_memory'
    
    return torch.device('cuda'), 'gpu_available'

# Fix matplotlib threading issues by setting backend before any plotting imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent threading issues
import matplotlib.pyplot as plt

# --- Project-specific modules ---
from utils.data_loader import load_power_system_data, create_data_loaders
from utils.metrics import PowerSystemLoss
from utils.data_validation import validate_data_before_training
from utils.optimization import (soa, setup_hyperparameter_bounds, create_model_kwargs, 
                               generate_run_name, process_optimization_params, 
                               calculate_objective_score)
from utils.evaluation import (evaluate_model, evaluate_moopf_objectives, 
                             save_best_model_results, print_comprehensive_summary,
                             print_model_summary)
from trainers.model_trainer import PowerSystemTrainer
from config import Config


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

def signal_handler(signum, frame):
    """Handle interrupt signals to ensure proper cleanup."""
    print(f"\nReceived signal {signum}. Cleaning up...")
    if _config_instance:
        _config_instance.finalize_run({'status': 'interrupted', 'reason': f'signal_{signum}'})
    sys.exit(0)

def main():
    global _config_instance
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Set PyTorch CUDA memory allocation configuration to prevent fragmentation
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("🔧 Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent memory fragmentation")
    
    class Args:
        # Configuration for models to test - now centralized in config
        test_config = 'sequential_only'  # Options: 'quick', 'core', 'comprehensive', 'physics_only', 'non_physics_only', 'sequential_only', 'all'
        bus_systems = 'all'  # Options: 'all', '33', '57', '118', or comma-separated like '33,57'
        seed = 42
        
        # === PARALLEL TRAINING CONFIGURATION ===
        # Device configuration
        force_cpu = False  # Set to True to force CPU training even if GPU is available
        
        # Parallel training mode
        parallel_data_loading = True   # Use multiple workers for data loading (recommended)
        
        # Worker configuration (auto-configured based on device if set to 'auto')
        data_workers = 'auto'         # Number of data loading workers
    
    args = Args()
    base_config = Config()
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
                        print(f"⚠️  Warning: {bus_num}-bus system not available. Available: {base_config.NUM_BUSES}")
                except ValueError:
                    print(f"⚠️  Warning: Invalid bus system '{bus_str}'. Skipping.")
            return bus_list if bus_list else base_config.NUM_BUSES
    
    bus_systems_to_test = parse_bus_systems(args.bus_systems)
    
    # Track all results for comprehensive summary
    all_results = []
    
    # STEP 1: Print run information
    run_info = base_config.get_run_info()
    print(f"\n🚀 STARTING NEW EXPERIMENTAL RUN")
    print(f"📅 Run ID: {run_info['run_id']}")
    print(f"⏰ Start Time: {run_info['start_time']}")
    print(f"📁 Results Directory: {run_info['current_run_dir']}")
    print(f"🔧 Test Configuration: {args.test_config}")
    print(f"🏭 Bus Systems to Test: {bus_systems_to_test}")
    print("="*80)
    
    # STEP 2: Validate data before training
    if not validate_data_before_training(base_config):
        print("❌ Data validation failed. Exiting training.")
        return
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # === DEVICE AND PARALLEL CONFIGURATION ===
    device, device_reason = get_safe_device(args.force_cpu, min_free_memory_gb=2.0)
    is_gpu = device.type == 'cuda'
    
    print(f"🔧 Using device: {device} (reason: {device_reason})")
    
    # Clear any existing GPU memory before starting
    clear_gpu_memory()
    
    # Auto-configure parallel settings based on device
    def get_optimal_workers():
        if is_gpu and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)")
            if gpu_memory >= 12:  # High-end GPU
                return {'data': 8}
            elif gpu_memory >= 8:  # Mid-range GPU  
                return {'data': 6}
            else:  # Entry-level GPU
                return {'data': 4}
        else:
            # CPU configuration
            try:
                import psutil
                cpu_count = psutil.cpu_count(logical=True)
                memory_gb = psutil.virtual_memory().total / (1024**3)
                print(f"🖥️  CPU: {cpu_count} cores, {memory_gb:.1f} GB RAM")
                if cpu_count >= 8 and memory_gb >= 16:
                    return {'data': 4}
                else:
                    return {'data': 4}
            except ImportError:
                return {'data': 4}
    
    optimal = get_optimal_workers()
    
    # Apply worker settings
    data_workers = optimal['data'] if args.data_workers == 'auto' else args.data_workers
    
    # Configure data loading
    if args.parallel_data_loading:
        base_config.NUM_WORKERS = data_workers
        print(f"📦 Parallel data loading: Enabled ({data_workers} workers)")
    else:
        base_config.NUM_WORKERS = 0
        print(f"📦 Parallel data loading: Disabled")
    
    print("="*80)

    # Get model configurations from config
    model_class_map = base_config.get_model_class_map()
    model_config_map = base_config.model_config_map
    models_to_test = base_config.get_models_to_test(args.test_config)


    # === MAIN TRAINING EXECUTION ===
    # Sequential bus system training (parallel bus systems disabled)
    print(f"\n🚀 SEQUENTIAL BUS SYSTEM TRAINING")
    print(f"🏭 Training {len(bus_systems_to_test)} bus systems: {bus_systems_to_test}")
    
    for num_buses in bus_systems_to_test:
        # Get adaptive MoSOA parameters for this system size
        mosoa_params = base_config._ModelConfig.get_adaptive_mosoa_params(num_buses)
        print(f"\n{'#'*80}\n# STARTING SEARCH FOR {num_buses}-BUS SYSTEM\n{'#'*80}")
        print(f"🎯 Optimization Strategy: {mosoa_params['strategy'].upper()}")
        print(f"📊 MoSOA Parameters: {mosoa_params['num_seagulls']} seagulls, {mosoa_params['max_iterations']} iterations")
        print(f"💡 Description: {mosoa_params['description']}")
        
        # Initialize data collectors for comparative plots
        bus_renewable_data = {}  # model_name -> renewable_impact_dataframe
        bus_convergence_data = {}  # model_name -> convergence_history
        all_tested_models = []  # Track all models tested (including non-physics)
        
        case_name = f"case{num_buses}"
        try:
            data_tuple = load_power_system_data(base_config, case_name)
            _features, _adjacency, _ybus_matrices, _targets, _energy_coeffs, _carbon_coeffs, _renewable_fractions, _normalizer = data_tuple
        except FileNotFoundError as e:
            print(f"[CRITICAL ERROR] {e}")
            continue

        for model_name in models_to_test:
            print(f"\n{'='*80}\nSTARTING HYPERPARAMETER SEARCH FOR: {model_name} on {num_buses}-bus\n{'='*80}")
            
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
                print(f"\n--- Evaluating {run_name} ---")

                try:
                    setup_logging(run_config.get_evaluation_path(f"{num_buses}bus/logs/{run_name}.log"))
                    
                    # Clear GPU memory before starting
                    clear_gpu_memory()
                    
                    # Check memory before model creation
                    if torch.cuda.is_available():
                        memory_info = check_gpu_memory()
                        print(f"GPU memory before model creation: {memory_info['allocated'] / 1024**3:.2f} GB allocated, {memory_info['free'] / 1024**3:.2f} GB free")
                    
                    # Use adaptive batch size based on system size
                    run_config.BATCH_SIZE = run_config.get_adaptive_batch_size(num_buses)
                    print(f"Using adaptive batch size: {run_config.BATCH_SIZE} for {num_buses}-bus system")
                    
                    loaders = create_data_loaders(
                        _features, _adjacency, _ybus_matrices, _targets, 
                        _energy_coeffs, _carbon_coeffs, _renewable_fractions, run_config, 
                        is_static=(not is_sequential)
                    )
                    train_loader, val_loader, test_loader = loaders

                    # Create model with optimized parameters
                    model_kwargs = create_model_kwargs(
                        model_config, params, num_buses, is_sequential, uses_adaptive_graph
                    )
                    
                    # Check memory before model creation
                    if torch.cuda.is_available():
                        memory_info = check_gpu_memory()
                        if memory_info['free'] < 1024**3:  # Less than 1GB free
                            print("WARNING: Low GPU memory, clearing cache...")
                            clear_gpu_memory()
                    
                    # Create model with error handling for OOM
                    try:
                        model = model_class_map[model_name](**model_kwargs).to(device)
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
                                is_static=(not is_sequential)
                            )
                            train_loader, val_loader, test_loader = loaders
                            model = model_class_map[model_name](**model_kwargs).to(device)
                        else:
                            raise e
                    
                    if torch.cuda.is_available():
                        memory_info = check_gpu_memory()
                        print(f"GPU memory after model creation: {memory_info['allocated'] / 1024**3:.2f} GB allocated, {memory_info['free'] / 1024**3:.2f} GB free")
                    
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
                    print(f"Training completed successfully for {run_name}")

                    # Get validation metrics for hyperparameter optimization
                    val_metrics = evaluate_model(model, val_loader, device, run_config, _normalizer, is_sequential)
                    
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
                            print(f"⚠️  Model state too large ({state_size / 1024**2:.1f} MB), not storing for memory efficiency")
                            model_state = None
                    except Exception as e:
                        print(f"⚠️  Could not save model state: {e}")
                        model_state = None
                    
                    run_results = {
                        'run_name': run_name, 
                        'model_name': model_name, 
                        **params, 
                        **test_metrics,  # Final test performance for reporting
                        'val_metrics': val_metrics,  # Validation metrics used for optimization
                        'total_loss': total_loss,  # Based on validation metrics
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
                    if 'model' in locals():
                        del model
                    if 'criterion' in locals():
                        del criterion
                    if 'optimizer' in locals():
                        del optimizer
                    if 'trainer' in locals():
                        del trainer
                    clear_gpu_memory()

            # Run MoSOA optimization
            best_score, best_position, history, iteration_details = soa(
                mosoa_params['num_seagulls'], 
                mosoa_params['max_iterations'], 
                lower_bounds, upper_bounds, dim, objective_function
            )

            # Process best parameters
            best_params = process_optimization_params(param_keys, best_position)

            print(f"\nBest hyperparameters found by MoSOA:")
            for key, value in best_params.items():
                print(f"{key}: {value}")
            print(f"Best score (total loss) achieved: {best_score:.6f}")

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
            model_kwargs_best = create_model_kwargs(
                model_config, best_params, num_buses, is_sequential, uses_adaptive_graph
            )

            # Create data loaders for best model
            loaders_best = create_data_loaders(
                _features, _adjacency, _ybus_matrices, _targets, 
                _energy_coeffs, _carbon_coeffs, _renewable_fractions, best_config, 
                is_static=(not is_sequential)
            )
            _, _, test_loader_best = loaders_best

            # Use the stored model state from the best run (if available)
            try:
                model_to_eval = model_class_map[model_name](**model_kwargs_best).to(device)
                if best_run.get('model_state') is not None:
                    model_to_eval.load_state_dict(best_run['model_state'])
                else:
                    print(f"⚠️  No model state available for {model_name}, using untrained model for evaluation")
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
                        is_static=(not is_sequential)
                    )
                    _, _, test_loader_best = loaders_best
                    model_to_eval = model_class_map[model_name](**model_kwargs_best).to(device)
                    model_to_eval.load_state_dict(best_run['model_state'])
                else:
                    raise e

            # Evaluate MOOPF objectives for the best model
            moopf_results, renewable_impact_data = evaluate_moopf_objectives(
                model_to_eval, test_loader_best, best_config, device, _normalizer, is_physics_informed
            )
            
            # Calculate final test performance metric for comparison
            if is_physics_informed:
                final_test_score = moopf_results['mse_score'].mean() if 'mse_score' in moopf_results.columns else best_run.get('mse', float('inf'))
                final_metric_name = "MOOPF MSE"
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
                'training_mse': best_run.get('mse', float('inf')),
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
            
            # Clear GPU cache after each model to prevent memory buildup
            clear_gpu_memory()
            print(f"🧹 GPU cache cleared after {model_name} training")
        
        # After all models for this bus system are complete, create comparative plots
        print(f"\n🎨 Creating comparative plots for {num_buses}-bus system...")
        
        # Import comparative visualization functions
        from utils.visualization import create_comparative_renewable_plots, create_comparative_convergence_plot
        
        # Create comparative renewable impact plots for all tested models
        # Always create plots if any models were tested, regardless of physics type
        if all_tested_models:
            try:
                create_comparative_renewable_plots(bus_renewable_data, base_config, num_buses, all_tested_models)
            except Exception as e:
                print(f"⚠️  Warning: Could not create renewable impact plots: {e}")
        
        # Create comparative convergence plot
        if bus_convergence_data:
            try:
                create_comparative_convergence_plot(bus_convergence_data, base_config, num_buses)
            except Exception as e:
                print(f"⚠️  Warning: Could not create convergence plots: {e}")
        
        # Final GPU cache clear after completing all models for this bus system
        clear_gpu_memory()
        print(f"🧹 GPU cache cleared after completing {num_buses}-bus system")
    
    # Print comprehensive final summary
    print_comprehensive_summary(all_results)
    
    # Finalize the run with summary
    if all_results:
        successful_results = [r for r in all_results if r['final_test_score'] != float('inf')]
        
        run_summary = {
            'models_tested': [r['model_name'] for r in all_results],
            'total_models': len(all_results),
            'successful_models': len(successful_results),
            'test_config': args.test_config,
            'best_model': successful_results[0]['model_name'] if successful_results else 'None',
            'best_score': successful_results[0]['final_test_score'] if successful_results else float('inf'),
            'bus_systems_tested': list(set(r['num_buses'] for r in all_results))
        }
        
        base_config.finalize_run(run_summary)
    else:
        base_config.finalize_run({'status': 'no_results', 'test_config': args.test_config})


if __name__ == '__main__':
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
    
    main()
