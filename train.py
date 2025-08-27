import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import copy
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import gc
import tempfile

# --- Project-specific modules ---
from models.adaptive_gcn import adaptiveGCN
from models.gcn import GCN
from models.pigcn import AdaptivePIGCN
from models.pigclstm import PIGCLSTM
from models.pigcgru import PIGCGRU
from models.ResnetPIGCGRU import ResnetPIGCGRU
from models.ResnetPIGCLSTM import ResnetPIGCLSTM
from utils.data_loader import load_power_system_data, create_data_loaders
from utils.metrics import PowerSystemLoss, compute_metrics
from utils.data_validation import validate_data_before_training
from trainers.model_trainer import PowerSystemTrainer
from config import Config
# --- End of imports ---


def setup_logging(log_path: str):
    """Initializes logging to both file and console."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_dir = os.path.dirname(log_path)
    if log_dir: os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path, mode='w'), logging.StreamHandler()])

def evaluate_model(model, test_loader, device, config, normalizer, is_sequential):
    """Evaluates the model on the test set and returns performance metrics."""
    model.eval()
    all_outputs, all_targets = [], []
    all_ybus = []  # Add this to collect Ybus matrices
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Evaluating {model.__class__.__name__}", leave=False)
        for batch in pbar:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            adj = batch['adjacency'].to(device)
            ybus = batch['ybus_matrix'].to(device)  # Get Ybus from batch

            # Handle sequential vs non-sequential models
            if is_sequential and features.dim() == 3:
                # For sequential models, use the last timestep
                features_input = features[:, -1, :]
            else:
                # For non-sequential models, use features as-is
                features_input = features
            
            outputs = model(features_input, adj)

            all_outputs.append(outputs)
            all_targets.append(targets)
            all_ybus.append(ybus)  # Store Ybus matrices

    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_ybus_tensor = torch.cat(all_ybus, dim=0)

    # Get num_buses dynamically from config without hardcoding
    if hasattr(config, 'NUM_BUSES'):
        num_buses = config.NUM_BUSES
        if isinstance(num_buses, list):
            num_buses = num_buses[0]  # Take first value if it's a list
    else:
        raise ValueError("Config must specify NUM_BUSES")
    
    outputs_denorm = normalizer.denormalize(all_outputs_tensor, num_buses)
    targets_denorm = normalizer.denormalize(all_targets_tensor, num_buses)

    return compute_metrics(outputs_denorm, targets_denorm, all_ybus_tensor, config)

def plot_training_history(history, model_name, config, num_buses):
    """Plots and saves the training history for the best model."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History for {model_name}', fontsize=16)

    # Plot total loss
    axes[0, 0].plot(history['train_total_loss'], label='Train')
    axes[0, 0].plot(history['val_total_loss'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot MSE
    axes[0, 1].plot(history['train_mse'], label='Train')
    axes[0, 1].plot(history['val_mse'], label='Validation')
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot power violation
    axes[1, 0].plot(history['train_power_violation'], label='Train')
    axes[1, 0].plot(history['val_power_violation'], label='Validation')
    axes[1, 0].set_title('Power Balance Violation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Violation')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot voltage violation
    axes[1, 1].plot(history['train_voltage_violation'], label='Train')
    axes[1, 1].plot(history['val_voltage_violation'], label='Validation')
    axes[1, 1].set_title('Voltage Violation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Violation')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save in the new directory structure
    save_path = config.get_training_history_path(num_buses, model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_renewable_impact(data_df, metric_name, y_label, title, config, num_buses, model_name):
    """Plots renewable impact for the best model."""
    x_col = 'renewable_fraction'
    y_col = metric_name
    
    plt.figure(figsize=(12, 8))
    x, y = data_df[x_col], data_df[y_col]
    plt.scatter(x, y, alpha=0.6, label='Test Scenario')

    # Fit trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x.sort_values(), p(x.sort_values()), "r--", linewidth=2, 
             label=f'Trendline (y={z[0]:.2f}x + {z[1]:.2f})')

    plt.title(title, fontsize=16)
    plt.xlabel('Renewable Energy Fraction', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Save in the new directory structure
    save_dir = config.get_renewable_impacts_dir(num_buses, model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{metric_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# Add this new function after the existing plotting functions
def plot_convergence(history, model_name, config, num_buses):
    """Plots the convergence curve of the MoSOA algorithm."""
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-', label='Convergence curve')
    plt.title(f'MoSOA Convergence for {model_name}', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best MSE + Physics-Informed Loss', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    save_path = config.get_convergence_plot_path(num_buses, model_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_best_model_results(best_model, best_run, moopf_results, renewable_impact_data, 
                          training_history, config, num_buses, is_physics_informed=True):
    """Saves all results for the best model in the new directory structure."""
    model_name = best_run['model_name']
    
    # Create necessary directories
    model_dir = config.get_model_eval_dir(num_buses, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model checkpoint
    torch.save(best_model.state_dict(), config.get_model_checkpoint_path(num_buses, model_name))
    
    # Save MOOPF results (or MSE results for non-physics models)
    results_filename = "moopf_results.csv" if is_physics_informed else "mse_results.csv"
    results_path = os.path.join(model_dir, results_filename)
    moopf_results.to_csv(results_path, index=False)
    
    # Save summary
    pd.DataFrame([best_run]).to_csv(config.get_summary_path(num_buses, model_name), index=False)
    
    # Plot training history (available for all models)
    plot_training_history(training_history, model_name, config, num_buses)
    
    # Plot convergence history if available (available for all models)
    if 'convergence_history' in best_run:
        plot_convergence(best_run['convergence_history'], model_name, config, num_buses)
    
    # Only plot renewable impacts for physics-informed models
    if is_physics_informed and not renewable_impact_data.empty:
        # Update metrics dictionary to match column names in renewable_impact_data
        metrics = {
            'normalized_carbon_emissions': 'Normalized Carbon Emissions',
            'voltage_deviation': 'Voltage Deviation',          # Changed from normalized_voltage_deviation
            'power_loss': 'Power Loss',                        # Changed from normalized_power_loss
            'power_flow': 'Normalized Power Flow'              # NEW: Added power flow metric
        }
        
        for metric, label in metrics.items():
            try:
                plot_renewable_impact(
                    renewable_impact_data,
                    metric_name=metric,
                    y_label=label,
                    title=f'Impact of Renewable Fraction on {label}',
                    config=config,
                    num_buses=num_buses,
                    model_name=model_name
                )
            except KeyError as e:
                print(f"Warning: Could not plot {metric} due to missing column: {e}")
                continue
    else:
        print(f"ℹ️  Skipping renewable impact plots for non-physics-informed model: {model_name}")

def evaluate_moopf_objectives(model, data_loader, config, device, normalizer, is_physics_informed=True):
    """Evaluates multi-objective objectives and collects data for analysis."""
    model.eval()
    num_buses_val = getattr(config, 'NUM_BUSES', 33)
    num_buses = int(num_buses_val[0]) if isinstance(num_buses_val, list) else int(num_buses_val)
    physics_calculator = PowerSystemLoss(config=config, normalizer=normalizer).to(device)
    w_loss, w_vdev, w_carbon = config.MOOPF_WEIGHT_LOSS, config.MOOPF_WEIGHT_VDEV, config.MOOPF_WEIGHT_CARBON
    all_results, renewable_impact_data = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating MOOPF Objectives"):
            features, ybus = batch['features'].to(device), batch['ybus_matrix'].to(device)
            time_carbon, time_energy = batch['time_carbon_coeffs'].to(device), batch['time_energy_coeffs'].to(device)
            adj = batch['adjacency'].to(device)

            outputs_norm = model(features, adj)
            outputs_phys = normalizer.denormalize(outputs_norm, num_buses)

            if is_physics_informed:
                # Calculate physics-based metrics for physics-informed models
                norm_loss = physics_calculator._compute_normalized_active_power_loss(outputs_phys, ybus)
                norm_vdev = physics_calculator._compute_normalized_voltage_deviation(outputs_phys)
                emissions = physics_calculator._compute_carbon_emissions(outputs_phys, time_carbon, time_energy)
                norm_power_flow = physics_calculator._compute_normalized_power_flow(outputs_phys, ybus)
            else:
                # For non-physics models, set physics metrics to zero/neutral values
                batch_size = features.shape[0]
                norm_loss = torch.zeros(batch_size, device=device)
                norm_vdev = torch.zeros(batch_size, device=device)
                emissions = {'raw': torch.zeros(batch_size, device=device), 'normalized': torch.zeros(batch_size, device=device)}
                norm_power_flow = torch.zeros(batch_size, device=device)

            # Capture data for analyzing the impact of renewables (only for physics-informed models)
            if is_physics_informed:
                try:
                    last_step_features = features[:, -1, ...] if features.dim() > 2 else features
                    inputs_phys = normalizer.denormalize(last_step_features, num_buses)
                    renewable_gen = inputs_phys[..., 4].sum(dim=-1)
                    total_load = inputs_phys[..., 2].sum(dim=-1) + 1e-9 # Add epsilon to avoid division by zero
                    renewable_fraction = (renewable_gen / total_load).cpu().numpy()

                    for i in range(features.shape[0]):
                        # --- START: MODIFICATION ---
                        # Using normalized carbon emissions for the plot and data capture
                        renewable_impact_data.append({
                            'renewable_fraction': renewable_fraction[i],
                            'normalized_carbon_emissions': emissions['normalized'][i].item(),
                            'voltage_deviation': norm_vdev[i].item(),
                            'power_loss': norm_loss[i].item(),
                            'power_flow': norm_power_flow[i].item()
                        })
                        # --- END: MODIFICATION ---
                except IndexError:
                    logging.warning("Could not calculate renewable fraction due to unexpected data shape.")

            if is_physics_informed:
                moopf_score = (w_loss * norm_loss + w_vdev * norm_vdev + w_carbon * emissions['normalized'])
                all_results.append({
                    'moopf_score': moopf_score.mean().item(), 'normalized_power_loss': norm_loss.mean().item(),
                    'normalized_voltage_deviation': norm_vdev.mean().item(), 'normalized_carbon_emissions': emissions['normalized'].mean().item(),
                    'raw_carbon_emissions_tCO2': emissions['raw'].mean().item(),
                    'normalized_power_flow': norm_power_flow.mean().item()  # Added but NOT in MOOPF score
                })
            else:
                # For non-physics models, only report MSE-based results
                mse_only = torch.mean((outputs_norm - normalizer.normalize(outputs_phys, num_buses))**2)
                all_results.append({
                    'mse_score': mse_only.item(),  # Main metric for non-physics models
                    'normalized_power_loss': 0.0,  # Not applicable
                    'normalized_voltage_deviation': 0.0,  # Not applicable
                    'normalized_carbon_emissions': 0.0,  # Not applicable
                    'raw_carbon_emissions_tCO2': 0.0,  # Not applicable
                    'normalized_power_flow': 0.0  # Not applicable
                })

    return pd.DataFrame(all_results), pd.DataFrame(renewable_impact_data)

def _init_positions(num_agents, dim, upper_bound, lower_bound):
    if isinstance(upper_bound, (int, float)): upper_bound = np.full(dim, upper_bound)
    if isinstance(lower_bound, (int, float)): lower_bound = np.full(dim, lower_bound)
    positions = np.zeros((num_agents, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lower_bound[i], upper_bound[i], num_agents)
    return positions

def soa(num_agents, max_iter, lower_bound, upper_bound, dim, objective_func):
    print("\nStarting Enhanced Seagull Optimization Algorithm for Hyperparameter Tuning...")
    best_position, best_score = np.zeros(dim), float('inf')
    positions = _init_positions(num_agents, dim, upper_bound, lower_bound)
    convergence_curve = []
    lambda_uncertainty, lambda_beta, beta_max = 5.0, 5.0, 2.0
    pbar = tqdm(range(max_iter), desc="MoSOA Progress")
    for l in pbar:
        fitness_all = [objective_func(np.clip(p, lower_bound, upper_bound)) for p in positions]
        valid_fitness = [(f, i) for i, f in enumerate(fitness_all) if f is not None and f != float('inf')]
        if valid_fitness:
            current_best_score_iter, best_agent_idx = min(valid_fitness, key=lambda item: item[0])
            if current_best_score_iter < best_score:
                best_score = current_best_score_iter
                best_position = positions[best_agent_idx].copy()
        convergence_curve.append(best_score)
        sigma = np.std([f for f, _ in valid_fitness]) if valid_fitness else 1e-9
        if sigma == 0: sigma = 1e-9
        fc, beta = 2 - l * (2 / max_iter), beta_max * np.exp(-lambda_beta * (l / max_iter))
        for i in range(num_agents):
            time_factor, uncertainty_factor = (1 - np.sin((np.pi / 2) * (l /    max_iter))), 1 / (1 + lambda_uncertainty * sigma)
            A1, b = 1.0 * time_factor * uncertainty_factor, 1.0 * (1 - 2 / (1 + np.exp((2 * l) / max_iter))) + -1.0
            rand_ll = (fc - 1) * np.random.rand() + 1

            D_alphs = fc * positions[i, :] + A1 * (best_position - positions[i, :])
            X1 = D_alphs * np.exp(b * rand_ll) * np.cos(rand_ll * 2 * np.pi) + best_position

            P_rand = positions[np.random.randint(0, num_agents), :]
            new_position = X1 + beta * (P_rand - positions[i, :])
            
            # CRITICAL FIX: Ensure positions stay within bounds
            positions[i, :] = np.clip(new_position, lower_bound, upper_bound)

        pbar.set_description(f"MoSOA Iteration {l+1}/{max_iter} | Best MSE: {best_score:.6f}")
    return best_score, best_position, convergence_curve

def main():
    class Args:
        # Updated model hierarchy: GCN -> adaptiveGCN -> PIGCN -> PIGCLSTM -> PIGCGRU -> ResnetPIGCLSTM -> ResnetPIGCGRU
        models_to_test = ['GCN', 'adaptiveGCN', 'PIGCN', 'PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU']
        seed = 42
        # MoSOA parameters are now adaptive - set dynamically based on system size
    args = Args()
    base_config = Config()
    
    # STEP 1: Validate data before training
    if not validate_data_before_training(base_config):
        print("❌ Data validation failed. Exiting training.")
        return
    
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = base_config.DEVICE

    model_class_map = {
        'adaptiveGCN': adaptiveGCN, 'GCN': GCN, 'PIGCN': AdaptivePIGCN, 'PIGCLSTM': PIGCLSTM,
        'PIGCGRU': PIGCGRU, 'ResnetPIGCGRU': ResnetPIGCGRU, 'ResnetPIGCLSTM': ResnetPIGCLSTM
    }
    model_config_map = {
        'GCN': base_config.GCNConfig, 'adaptiveGCN': base_config.adaptiveGCNConfig, 'PIGCN': base_config.PIGCNConfig,
        'PIGCLSTM': base_config.PIGCLSTMConfig, 'PIGCGRU': base_config.PIGCGRUConfig,
        'ResnetPIGCGRU': base_config.ResnetPIGCGRUConfig, 'ResnetPIGCLSTM': base_config.ResnetPIGCLSTMConfig
    }

    bus_systems_to_test = base_config.NUM_BUSES if isinstance(base_config.NUM_BUSES, list) else [base_config.NUM_BUSES]

    for num_buses in bus_systems_to_test:
        # Get adaptive MoSOA parameters for this system size
        mosoa_params = base_config._ModelConfig.get_adaptive_mosoa_params(num_buses)
        print(f"\n{'#'*80}\n# STARTING SEARCH FOR {num_buses}-BUS SYSTEM\n{'#'*80}")
        print(f"🎯 Optimization Strategy: {mosoa_params['strategy'].upper()}")
        print(f"📊 MoSOA Parameters: {mosoa_params['num_seagulls']} seagulls, {mosoa_params['max_iterations']} iterations")
        print(f"💡 Description: {mosoa_params['description']}")
        
        all_bus_system_results = []
        case_name = f"case{num_buses}"
        try:
            data_tuple = load_power_system_data(base_config, case_name)
            _features, _adjacency, _ybus_matrices, _targets, _energy_coeffs, _carbon_coeffs, _normalizer = data_tuple
        except FileNotFoundError as e:
            print(f"[CRITICAL ERROR] {e}"); continue

        for model_name in args.models_to_test:
            print(f"\n{'='*80}\nSTARTING HYPERPARAMETER SEARCH FOR: {model_name} on {num_buses}-bus\n{'='*80}")
            model_specific_results, model_config = [], model_config_map[model_name]

            is_sequential = 'LSTM' in model_name.upper() or 'GRU' in model_name.upper()
            is_physics_informed = 'PI' in model_name  # Only models with 'PI' prefix are physics-informed
            uses_adaptive_graph = model_name in ['PIGCLSTM', 'PIGCGRU', 'adaptiveGCN', 'PIGCN', 'ResnetPIGCGRU', 'ResnetPIGCLSTM']

            # Use adaptive scaling for hidden dimensions based on system size
            hidden_range = model_config.get_hidden_dim_range(num_buses) if hasattr(model_config, 'get_hidden_dim_range') else model_config.HIDDEN_DIM_RANGE
            param_bounds = {'HIDDEN_DIM': hidden_range, 'NUM_GC_LAYERS': model_config.NUM_GC_LAYERS_RANGE}
            if is_physics_informed: 
                # By adding these lines, we tell MoSoa to tune these values.
                param_bounds['LAMBDA_P'] = (1.0, 50.0)
                param_bounds['LAMBDA_V'] = (1.0, 50.0)
            if is_sequential: param_bounds.update({'SEQUENCE_LENGTH': model_config.SEQUENCE_LENGTH_RANGE, 'RNN_LAYERS': model_config.RNN_LAYERS_RANGE})
            if uses_adaptive_graph: param_bounds.update({'EMBEDDING_DIM': model_config.EMBEDDING_DIM_RANGE, 'PHI': model_config.PHI_RANGE})

            param_keys, dim, lower_bounds, upper_bounds = list(param_bounds.keys()), len(param_bounds), [b[0] for b in param_bounds.values()], [b[1] for b in param_bounds.values()]

            def objective_function(params_array):
                params = {key: val for key, val in zip(param_keys, params_array)}
                for k in ['HIDDEN_DIM', 'NUM_GC_LAYERS', 'SEQUENCE_LENGTH', 'RNN_LAYERS', 'EMBEDDING_DIM']:
                    if k in params: params[k] = int(round(params[k]))

                run_config = copy.deepcopy(base_config)
                for key, value in params.items(): 
                    setattr(run_config, key.upper(), value)
                run_config.NUM_BUSES = num_buses

                run_name = f"run_{model_name}_B{num_buses}_H{params.get('HIDDEN_DIM', 'N/A')}_GC{params.get('NUM_GC_LAYERS', 'N/A')}"
                if is_sequential: 
                    run_name += f"_SL{params.get('SEQUENCE_LENGTH', 'N/A')}_R{params.get('RNN_LAYERS', 'N/A')}"
                print(f"\n--- Evaluating {run_name} ---")

                try:
                    setup_logging(run_config.get_evaluation_path(f"{num_buses}bus/logs/{run_name}.log"))
                    loaders = create_data_loaders(
                        _features, _adjacency, _ybus_matrices, _targets, 
                        _energy_coeffs, _carbon_coeffs, run_config, 
                        is_static=(not is_sequential)
                    )
                    train_loader, val_loader, test_loader = loaders

                    model_kwargs = {
                        'feature_dim': model_config.FEATURE_DIM,
                        'hidden_dim': params['HIDDEN_DIM'],
                        'num_gc_layers': params['NUM_GC_LAYERS'],
                        'num_buses': num_buses,
                        'dropout': model_config.DROPOUT
                    }
                    if is_sequential: 
                        model_kwargs['rnn_layers'] = params['RNN_LAYERS']
                    if uses_adaptive_graph: 
                        model_kwargs.update({
                            'embedding_dim': params['EMBEDDING_DIM'],
                            'phi': params['PHI']
                        })

                    model = model_class_map[model_name](**model_kwargs).to(device)
                    
                    # Use appropriate loss function based on whether model is physics-informed
                    if is_physics_informed:
                        criterion = PowerSystemLoss(config=run_config, normalizer=_normalizer).to(device)
                    else:
                        # For non-physics-informed models, use simple MSE loss
                        criterion = PowerSystemLoss(config=run_config, normalizer=_normalizer, is_gcn=True).to(device)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.LEARNING_RATE)

                    trainer = PowerSystemTrainer(model, criterion, optimizer, run_config, device)
                    trainer.train(train_loader, val_loader)

                    # Get validation metrics for hyperparameter optimization (NOT test!)
                    val_metrics = evaluate_model(model, val_loader, device, run_config, _normalizer, is_sequential)
                    
                    # Get test metrics for final evaluation
                    test_metrics = evaluate_model(model, test_loader, device, run_config, _normalizer, is_sequential)

                    # Calculate total loss for optimization using VALIDATION metrics
                    if is_physics_informed:
                        total_loss = (val_metrics['mse'] + 
                                    run_config.LAMBDA_P * val_metrics['power_violation'] + 
                                    run_config.LAMBDA_V * val_metrics['voltage_violation'])
                    else:
                        # For non-physics-informed models, only use MSE
                        total_loss = val_metrics['mse']

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
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # After SOA returns results - use adaptive parameters
            best_score, best_position, history = soa(
                mosoa_params['num_seagulls'], 
                mosoa_params['max_iterations'], 
                lower_bounds, upper_bounds, dim, objective_function
            )

            # Create dictionary of best parameters found by MoSOA
            best_params = {key: val for key, val in zip(param_keys, best_position)}
            for k in ['HIDDEN_DIM', 'NUM_GC_LAYERS', 'SEQUENCE_LENGTH', 'RNN_LAYERS', 'EMBEDDING_DIM']:
                if k in best_params:
                    best_params[k] = int(round(best_params[k]))

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

            # Get the best run based on total_loss and add MoSOA results
            best_run = best_run_df.loc[best_run_df['total_loss'].idxmin()].to_dict()
            best_run.update({
                'convergence_history': history,
                'mosoa_best_score': best_score,
                'mosoa_best_params': best_params
            })

            # Create best config from the best parameters
            best_config = copy.deepcopy(base_config)
            for key, value in best_params.items():
                setattr(best_config, key.upper(), value)
            best_config.NUM_BUSES = num_buses

            # Create model kwargs for best model
            best_model_config = model_config_map[model_name]
            model_kwargs_best = {
                'feature_dim': best_model_config.FEATURE_DIM,
                'hidden_dim': int(best_params['HIDDEN_DIM']),
                'num_gc_layers': int(best_params['NUM_GC_LAYERS']),
                'num_buses': num_buses,
                'dropout': best_model_config.DROPOUT
            }
            if is_sequential:
                model_kwargs_best['rnn_layers'] = int(best_params['RNN_LAYERS'])
            if uses_adaptive_graph:
                model_kwargs_best.update({
                    'embedding_dim': int(best_params['EMBEDDING_DIM']),
                    'phi': float(best_params['PHI'])
                })

            # Create data loaders for best model
            loaders_best = create_data_loaders(
                _features, _adjacency, _ybus_matrices, _targets, 
                _energy_coeffs, _carbon_coeffs, best_config, 
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
            
            if is_physics_informed:
                print("\n--- MOOPF Evaluation ---", moopf_results.mean().to_dict(), sep='\n')
            else:
                print("\n--- MSE Evaluation ---", moopf_results.mean().to_dict(), sep='\n')

            # Save all results using the training history from the best run
            save_best_model_results(
                best_model=model_to_eval,
                best_run=best_run,
                moopf_results=moopf_results,
                renewable_impact_data=renewable_impact_data,
                training_history=best_run['training_history'],
                config=best_config,
                num_buses=num_buses,
                is_physics_informed=is_physics_informed
            )
    

if __name__ == '__main__':
    main()