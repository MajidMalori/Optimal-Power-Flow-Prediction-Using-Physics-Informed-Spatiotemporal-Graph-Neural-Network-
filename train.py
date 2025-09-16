import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import gc
import signal
import time

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


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Training timeout exceeded")


def run_with_timeout(func, timeout_seconds=600):  # 10 minutes default timeout
    """Run a function with a timeout."""
    if hasattr(signal, 'SIGALRM'):  # Unix systems
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            result = func()
            return result
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:  # Windows systems - use threading timeout
        import threading
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        
        if thread.is_alive():
            raise TimeoutError(f"Training timeout exceeded ({timeout_seconds} seconds)")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]


def main():
    class Args:
        # Configuration for models to test - now centralized in config
        test_config = 'all'  # Options: 'quick', 'comprehensive', 'physics_only', 'non_physics_only', 'all'
        seed = 42
    
    args = Args()
    base_config = Config()
    
    # Track all results for comprehensive summary
    all_results = []
    
    # STEP 1: Print run information
    run_info = base_config.get_run_info()
    print(f"\n🚀 STARTING NEW EXPERIMENTAL RUN")
    print(f"📅 Run ID: {run_info['run_id']}")
    print(f"⏰ Start Time: {run_info['start_time']}")
    print(f"📁 Results Directory: {run_info['current_run_dir']}")
    print(f"🔧 Test Configuration: {args.test_config}")
    print("="*80)
    
    # STEP 2: Validate data before training
    if not validate_data_before_training(base_config):
        print("❌ Data validation failed. Exiting training.")
        return
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = base_config.DEVICE

    # Get model configurations from config
    model_class_map = base_config.get_model_class_map()
    model_config_map = base_config.model_config_map
    models_to_test = base_config.get_models_to_test(args.test_config)

    bus_systems_to_test = (base_config.NUM_BUSES 
                          if isinstance(base_config.NUM_BUSES, list) 
                          else [base_config.NUM_BUSES])

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
                    
                    # CRITICAL FIX: Clear GPU memory before each run
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print(f"GPU memory before model creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    
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
                    
                    # CRITICAL FIX: Add memory check before model creation
                    if torch.cuda.is_available():
                        available_memory = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                        print(f"Available GPU memory: {available_memory / 1024**3:.2f} GB")
                        if available_memory < 1024**3:  # Less than 1GB
                            print("WARNING: Low GPU memory, clearing cache...")
                            torch.cuda.empty_cache()
                    
                    model = model_class_map[model_name](**model_kwargs).to(device)
                    
                    # CRITICAL FIX: Check memory after model creation
                    if torch.cuda.is_available():
                        print(f"GPU memory after model creation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    
                    # Use appropriate loss function based on whether model is physics-informed
                    criterion = PowerSystemLoss(
                        config=run_config, 
                        normalizer=_normalizer, 
                        is_gcn=(not is_physics_informed)
                    ).to(device)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.LEARNING_RATE)

                    trainer = PowerSystemTrainer(model, criterion, optimizer, run_config, device, is_physics_informed)
                    
                    # CRITICAL FIX: Add timeout protection for training
                    def train_model():
                        trainer.train(train_loader, val_loader)
                        return True
                    
                    try:
                        # Set timeout based on system size (larger systems need more time)
                        timeout_seconds = 300 if num_buses <= 33 else 600 if num_buses <= 57 else 900
                        run_with_timeout(train_model, timeout_seconds)
                        print(f"Training completed successfully for {run_name}")
                    except TimeoutError as e:
                        logging.error(f"Training timeout for {run_name}: {e}")
                        return float('inf')

                    # Get validation metrics for hyperparameter optimization
                    val_metrics = evaluate_model(model, val_loader, device, run_config, _normalizer, is_sequential)
                    
                    # Get test metrics for final evaluation
                    test_metrics = evaluate_model(model, test_loader, device, run_config, _normalizer, is_sequential)

                    # Calculate total loss for optimization using validation metrics
                    total_loss = calculate_objective_score(val_metrics, run_config, is_physics_informed)

                    # Store the training history with the results
                    run_results = {
                        'run_name': run_name, 
                        'model_name': model_name, 
                        **params, 
                        **test_metrics,  # Final test performance for reporting
                        'val_metrics': val_metrics,  # Validation metrics used for optimization
                        'total_loss': total_loss,  # Based on validation metrics
                        'training_history': trainer.get_training_history(),
                        'model_state': model.state_dict(),
                        'model_config': run_config  
                    }
                    model_specific_results.append(run_results)

                    return total_loss
                    
                except Exception as e:
                    logging.error(f"Run {run_name} failed: {e}", exc_info=True)
                    return float('inf')
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

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

            # Use the stored model state from the best run
            model_to_eval = model_class_map[model_name](**model_kwargs_best).to(device)
            model_to_eval.load_state_dict(best_run['model_state'])

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
    main()
