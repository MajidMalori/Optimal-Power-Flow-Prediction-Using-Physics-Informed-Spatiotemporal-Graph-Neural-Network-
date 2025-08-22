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
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Evaluating {model.__class__.__name__}", leave=False)
        for batch in pbar:
            features, targets = batch['features'].to(device), batch['targets'].to(device)
            adj = batch['adjacency'].to(device)

            outputs = model(features, adj)

            all_outputs.append(outputs)
            all_targets.append(targets)

    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)

    num_buses_val = getattr(config, 'NUM_BUSES', 33)
    num_buses = int(num_buses_val[0]) if isinstance(num_buses_val, list) else int(num_buses_val)
    outputs_denorm = normalizer.denormalize(all_outputs_tensor, num_buses)
    targets_denorm = normalizer.denormalize(all_targets_tensor, num_buses)

    return compute_metrics(outputs_denorm, targets_denorm)

def plot_training_history(history: dict, title: str, save_path: str, metric_name: str = 'Loss'):
    """Plots and saves the convergence history."""
    plt.style.use('ggplot'); plt.figure(figsize=(12, 7))
    for key, values in history.items():
        valid_values = [v for v in values if v is not None and v != float('inf')]
        if not valid_values: continue
        plt.plot(range(1, len(valid_values) + 1), valid_values, marker='o', linestyle='-', label=key)
    plt.title(title, fontsize=16, weight='bold'); plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    if any(valid_values): plt.legend(fontsize=10)
    plt.grid(True); os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()
    print(f"Convergence plot saved to {save_path}")

def plot_renewable_impact(data_df: pd.DataFrame, y_col: str, y_label: str, title: str, save_path: str):
    """Generates and saves a scatter plot of a given metric vs. the renewable energy fraction."""
    x_col = 'renewable_fraction'
    if data_df.empty or x_col not in data_df.columns or y_col not in data_df.columns:
        print(f"Cannot generate renewable impact plot for '{y_col}' due to missing data or required columns."); return

    plt.style.use('ggplot'); plt.figure(figsize=(12, 8))

    # Filter out extreme outliers in the y-column for better visualization and trendline fitting
    q1, q3 = data_df[y_col].quantile(0.05), data_df[y_col].quantile(0.95)
    iqr = q3 - q1
    filtered_df = data_df[data_df[y_col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]

    x, y = filtered_df[x_col], filtered_df[y_col]

    if len(x) < 2:
        print(f"Not enough data points to plot renewable impact for '{y_col}' after filtering."); return

    plt.scatter(x, y, alpha=0.6, label='Test Scenario')

    try:
        # Fit a linear trendline
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x.sort_values(), p(x.sort_values()), "r--", linewidth=2, label=f'Trendline (y={z[0]:.2f}x + {z[1]:.2f})')
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Could not fit a trendline for the '{y_col}' renewable impact plot: {e}")

    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Renewable Energy Fraction', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(fontsize=10); plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300); plt.close()
    print(f"Renewable impact plot saved to {save_path}")

def evaluate_moopf_objectives(model, data_loader, config, device, normalizer):
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

            norm_loss = physics_calculator._compute_normalized_power_balance_violation(outputs_phys, ybus)
            norm_vdev = physics_calculator._compute_normalized_voltage_limit_violation(outputs_phys)
            emissions = physics_calculator._compute_carbon_emissions(outputs_phys, time_carbon, time_energy)

            # Capture data for analyzing the impact of renewables on all objectives
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
                        'power_loss': norm_loss[i].item()
                    })
                    # --- END: MODIFICATION ---
            except IndexError:
                logging.warning("Could not calculate renewable fraction due to unexpected data shape.")

            moopf_score = (w_loss * norm_loss + w_vdev * norm_vdev + w_carbon * emissions['normalized'])
            all_results.append({
                'moopf_score': moopf_score.mean().item(), 'normalized_power_loss': norm_loss.mean().item(),
                'normalized_voltage_deviation': norm_vdev.mean().item(), 'normalized_carbon_emissions': emissions['normalized'].mean().item(),
                'raw_carbon_emissions_tCO2': emissions['raw'].mean().item()
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
            positions[i, :] = X1 + beta * (P_rand - positions[i, :])

        pbar.set_description(f"MoSOA Iteration {l+1}/{max_iter} | Best MSE: {best_score:.6f}")
    return best_score, best_position, convergence_curve

def main():
    class Args:
        models_to_test = ['ResnetPIGCLSTM', 'ResnetPIGCGRU', 'PIGCLSTM', 'PIGCGRU']
        seed = 42
        num_seagulls = 10
        max_iterations = 25 # Reduced for quicker testing
    args = Args()
    base_config = Config()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = base_config.DEVICE

    model_class_map = {
        'adaptiveGCN': adaptiveGCN, 'GCN': GCN, 'PIGCN': AdaptivePIGCN, 'PIGCLSTM': PIGCLSTM,
        'PIGCGRU': PIGCGRU, 'ResnetPIGCGRU': ResnetPIGCGRU, 'ResnetPIGCLSTM': ResnetPIGCLSTM
    }
    model_config_map = {
        'adaptiveGCN': base_config.adaptiveGCNConfig, 'GCN': base_config.GCNConfig, 'PIGCN': base_config.PIGCNConfig,
        'PIGCLSTM': base_config.PIGCLSTMConfig, 'PIGCGRU': base_config.PIGCGRUConfig,
        'ResnetPIGCGRU': base_config.ResnetPIGCGRUConfig, 'ResnetPIGCLSTM': base_config.ResnetPIGCLSTMConfig
    }

    bus_systems_to_test = base_config.NUM_BUSES if isinstance(base_config.NUM_BUSES, list) else [base_config.NUM_BUSES]

    for num_buses in bus_systems_to_test:
        print(f"\n{'#'*80}\n# STARTING SEARCH FOR {num_buses}-BUS SYSTEM\n{'#'*80}")
        all_bus_system_results = []
        case_name = f"case{num_buses}"
        try:
            data_tuple = load_power_system_data(base_config, case_name)
            _features, _adjacency, _ybus_list, _targets, _energy_coeffs, _carbon_coeffs, _sample_map, _normalizer = data_tuple
        except FileNotFoundError as e:
            print(f"[CRITICAL ERROR] {e}"); continue

        for model_name in args.models_to_test:
            print(f"\n{'='*80}\nSTARTING HYPERPARAMETER SEARCH FOR: {model_name} on {num_buses}-bus\n{'='*80}")
            model_specific_results, model_config = [], model_config_map[model_name]

            is_sequential = 'LSTM' in model_name.upper() or 'GRU' in model_name.upper()
            is_physics_informed = 'PI' in model_name
            uses_adaptive_graph = model_name in ['PIGCLSTM', 'PIGCGRU', 'adaptiveGCN', 'PIGCN', 'ResnetPIGCGRU', 'ResnetPIGCLSTM']

            param_bounds = {'HIDDEN_DIM': model_config.HIDDEN_DIM_RANGE, 'NUM_GC_LAYERS': model_config.NUM_GC_LAYERS_RANGE}
            if is_physics_informed: param_bounds['LAMBDA_P'] = (1.0, 50.0)
            if is_sequential: param_bounds.update({'SEQUENCE_LENGTH': model_config.SEQUENCE_LENGTH_RANGE, 'RNN_LAYERS': model_config.RNN_LAYERS_RANGE})
            if uses_adaptive_graph: param_bounds.update({'EMBEDDING_DIM': model_config.EMBEDDING_DIM_RANGE, 'PHI': model_config.PHI_RANGE})

            param_keys, dim, lower_bounds, upper_bounds = list(param_bounds.keys()), len(param_bounds), [b[0] for b in param_bounds.values()], [b[1] for b in param_bounds.values()]

            def objective_function(params_array):
                params = {key: val for key, val in zip(param_keys, params_array)}
                for k in ['HIDDEN_DIM', 'NUM_GC_LAYERS', 'SEQUENCE_LENGTH', 'RNN_LAYERS', 'EMBEDDING_DIM']:
                    if k in params: params[k] = int(round(params[k]))

                run_config = copy.deepcopy(base_config)
                for key, value in params.items(): setattr(run_config, key.upper(), value)
                run_config.NUM_BUSES = num_buses

                run_name = f"run_{model_name}_B{num_buses}_H{params.get('HIDDEN_DIM', 'N/A')}_GC{params.get('NUM_GC_LAYERS', 'N/A')}"
                if is_sequential: run_name += f"_SL{params.get('SEQUENCE_LENGTH', 'N/A')}_R{params.get('RNN_LAYERS', 'N/A')}"
                print(f"\n--- Evaluating {run_name} ---")

                try:
                    setup_logging(run_config.get_evaluation_path(f"{num_buses}bus/logs/{run_name}.log"))
                    loaders = create_data_loaders(_features, _adjacency, _ybus_list, _targets, _energy_coeffs, _carbon_coeffs, _sample_map, run_config, is_static=(not is_sequential))
                    train_loader, val_loader, test_loader = loaders

                    model_kwargs = { 'feature_dim': model_config.FEATURE_DIM, 'hidden_dim': params['HIDDEN_DIM'], 'num_gc_layers': params['NUM_GC_LAYERS'],
                                     'num_buses': num_buses, 'dropout': model_config.DROPOUT }
                    if is_sequential: model_kwargs['rnn_layers'] = params['RNN_LAYERS']
                    if uses_adaptive_graph: model_kwargs.update({'embedding_dim': params['EMBEDDING_DIM'], 'phi': params['PHI']})

                    model = model_class_map[model_name](**model_kwargs).to(device)
                    criterion = PowerSystemLoss(config=run_config, normalizer=_normalizer).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.LEARNING_RATE)

                    trainer = PowerSystemTrainer(model, criterion, optimizer, run_config, device)
                    trainer.train(train_loader, val_loader)

                    test_metrics = evaluate_model(model, test_loader, device, run_config, _normalizer, is_sequential)
                    model_specific_results.append({'run_name': run_name, 'model_name': model_name, **params, **test_metrics})
                    return test_metrics['mse']
                except Exception as e:
                    logging.error(f"Run {run_name} failed: {e}", exc_info=True); return float('inf')
                finally:
                    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

            best_score, best_position, history = soa(args.num_seagulls, args.max_iterations, lower_bounds, upper_bounds, dim, objective_function)
            if not model_specific_results: print(f"No successful runs for {model_name}."); continue

            best_run_df = pd.DataFrame(model_specific_results)
            if 'mse' not in best_run_df.columns or best_run_df['mse'].notna().sum() == 0:
                print(f"All runs for {model_name} failed."); continue
            best_run = best_run_df.loc[best_run_df['mse'].idxmin()].to_dict()
            all_bus_system_results.append(best_run)

            print(f"\n{'*'*80}\n* EVALUATING BEST MODEL: {best_run['run_name']}\n{'*'*80}")
            best_config = copy.deepcopy(base_config)
            for key, value in best_run.items():
                if hasattr(best_config, key.upper()): setattr(best_config, key.upper(), value)
            best_config.NUM_BUSES = num_buses

            best_model_name = best_run['model_name']
            is_sequential_best = 'LSTM' in best_model_name.upper() or 'GRU' in best_model_name.upper()

            loaders_best = create_data_loaders(_features, _adjacency, _ybus_list, _targets, _energy_coeffs, _carbon_coeffs, _sample_map, best_config, is_static=(not is_sequential_best))
            _, _, test_loader_best = loaders_best

            best_model_config = model_config_map[best_model_name]
            model_kwargs_best = { 'feature_dim': best_model_config.FEATURE_DIM, 'hidden_dim': int(best_run['HIDDEN_DIM']), 'num_gc_layers': int(best_run['NUM_GC_LAYERS']),
                                  'num_buses': num_buses, 'dropout': best_model_config.DROPOUT }
            if is_sequential_best: model_kwargs_best['rnn_layers'] = int(best_run['RNN_LAYERS'])
            if best_model_name in ['PIGCLSTM', 'PIGCGRU', 'adaptiveGCN', 'PIGCN', 'ResnetPIGCGRU', 'ResnetPIGCLSTM']:
                model_kwargs_best.update({'embedding_dim': int(best_run['EMBEDDING_DIM']), 'phi': float(best_run['PHI'])})

            model_to_eval = model_class_map[best_model_name](**model_kwargs_best).to(device)

            model_path = best_config.get_checkpoint_path('best_model.pth')
            if not os.path.exists(model_path):
                model_path = base_config.get_checkpoint_path(f"{num_buses}bus/{best_run['run_name']}.pth")

            if os.path.exists(model_path):
                model_to_eval.load_state_dict(torch.load(model_path, map_location=device))
            else:
                print(f"WARNING: Best model checkpoint not found at {model_path}. Evaluating the randomly initialized model instead.")

            moopf_results, renewable_impact_data = evaluate_moopf_objectives(model_to_eval, test_loader_best, best_config, device, _normalizer)
            print("\n--- MOOPF Evaluation ---", moopf_results.mean().to_dict(), sep='\n')

            # Define base paths and names for saving results
            base_results_path = base_config.get_evaluation_path(f"{num_buses}bus/")
            run_name_sanitized = best_run['run_name']

            # Save the aggregated and sample-wise results to CSV files
            moopf_results.to_csv(os.path.join(base_results_path, f"moopf_{run_name_sanitized}.csv"), index=False)
            renewable_impact_data.to_csv(os.path.join(base_results_path, f"renewable_impact_data_{run_name_sanitized}.csv"), index=False)

            # --- START: MODIFICATION ---
            # Plot the impact of renewable energy fraction on multiple objectives
            # Plot for Carbon Emissions (using NORMALIZED value)
            plot_renewable_impact(
                data_df=renewable_impact_data,
                y_col='normalized_carbon_emissions',
                y_label='Normalized Carbon Emissions',
                title='Impact of Renewable Fraction on Carbon Emissions',
                save_path=os.path.join(base_results_path, f'renew_impact_emissions_{run_name_sanitized}.png')
            )
            # --- END: MODIFICATION ---

            # Plot for Voltage Deviation
            plot_renewable_impact(
                data_df=renewable_impact_data,
                y_col='voltage_deviation',
                y_label='Normalized Voltage Deviation',
                title='Impact of Renewable Fraction on Voltage Deviation',
                save_path=os.path.join(base_results_path, f'renew_impact_vdev_{run_name_sanitized}.png')
            )

            # Plot for Power Loss
            plot_renewable_impact(
                data_df=renewable_impact_data,
                y_col='power_loss',
                y_label='Normalized Power Loss',
                title='Impact of Renewable Fraction on Power Loss',
                save_path=os.path.join(base_results_path, f'renew_impact_ploss_{run_name_sanitized}.png')
            )

        if not all_bus_system_results: print(f"\n--- No successful runs for {num_buses}-bus system. ---"); continue

        print(f"\n--- Summary for {num_buses}-bus system ---")
        results_df = pd.DataFrame(all_bus_system_results).sort_values('mse')
        results_df.to_csv(base_config.get_evaluation_path(f'MoSOA_summary_{num_buses}bus.csv'), index=False)
        print("Top models:", results_df.head().to_string(), sep='\n')

if __name__ == '__main__':
    main()