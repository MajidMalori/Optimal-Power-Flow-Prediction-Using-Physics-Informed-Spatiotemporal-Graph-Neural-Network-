# File: train.py

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
import seaborn as sns

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

def plot_renewable_impact(data_df: pd.DataFrame, y_col: str, title: str, y_label: str, save_path: str):
    """Generates and saves a scatter plot of a metric vs. renewable fraction."""
    x_col = 'renewable_fraction'
    if data_df.empty or x_col not in data_df.columns or y_col not in data_df.columns:
        print(f"Cannot generate plot '{title}' due to missing data ('{x_col}' or '{y_col}')."); return
    plt.style.use('ggplot'); plt.figure(figsize=(12, 8))
    q1, q3 = data_df[y_col].quantile(0.05), data_df[y_col].quantile(0.95)
    iqr = q3 - q1
    filtered_df = data_df[data_df[y_col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
    x, y = filtered_df[x_col], filtered_df[y_col]
    if len(x) < 2:
        print(f"Not enough data points to plot '{title}' after filtering."); return
    plt.scatter(x, y, alpha=0.6, label='Test Scenario')
    try:
        z = np.polyfit(x, y, 1); p = np.poly1d(z)
        plt.plot(x, p(x), "r--", linewidth=2, label=f'Trendline (y={z[0]:.2f}x + {z[1]:.2f})')
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Could not fit a trendline for the plot '{title}': {e}")
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Renewable Energy Fraction', fontsize=12); plt.ylabel(y_label, fontsize=12)
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
            norm_loss_per_instance = physics_calculator._compute_normalized_power_balance_violation(outputs_phys, ybus)
            norm_vdev_per_instance = physics_calculator._compute_normalized_voltage_limit_violation(outputs_phys)
            emissions = physics_calculator._compute_carbon_emissions(outputs_phys, time_carbon, time_energy)
            try:
                last_step_features = features[:, -1, ...] if features.dim() > 2 else features
                inputs_phys = normalizer.denormalize(last_step_features, num_buses)
                renewable_gen = inputs_phys[..., 4].sum(dim=-1)
                total_load = inputs_phys[..., 2].sum(dim=-1) + 1e-9
                renewable_fraction = (renewable_gen / total_load).cpu().numpy()
                for i in range(features.shape[0]):
                    renewable_impact_data.append({
                        'renewable_fraction': renewable_fraction[i], 'carbon_emissions': emissions['raw'][i].item(),
                        'normalized_power_loss': norm_loss_per_instance[i].item(),
                        'normalized_voltage_deviation': norm_vdev_per_instance[i].item()
                    })
            except IndexError:
                logging.warning("Could not calculate renewable fraction.")
            moopf_score = (w_loss * norm_loss_per_instance.mean() + w_vdev * norm_vdev_per_instance.mean() + w_carbon * emissions['normalized'].mean())
            all_results.append({
                'moopf_score': moopf_score.item(), 'normalized_power_loss': norm_loss_per_instance.mean().item(),
                'normalized_voltage_deviation': norm_vdev_per_instance.mean().item(),
                'normalized_carbon_emissions': emissions['normalized'].mean().item(),
                'raw_carbon_emissions_tCO2': emissions['raw'].mean().item()
            })
    return pd.DataFrame(all_results), pd.DataFrame(renewable_impact_data)

def plot_MoSoA_performance(df: pd.DataFrame, config: Config):
    """
    Plots the impact of MoSoA parameters (iterations, seagulls) on the best MSE score.
    """
    if df.empty:
        print("Tuning results DataFrame is empty. Skipping performance plot.")
        return
    g = sns.relplot(
        data=df, x='max_iterations', y='best_mse', hue='num_seagulls',
        col='model_name', row='num_buses', kind='line', marker='o',
        palette='viridis', height=5, aspect=1.2, facet_kws={'margin_titles': True}
    )
    g.set_axis_labels("Max Iterations", "Best MSE Score")
    g.set_titles(col_template="{col_name}", row_template="{row_name}-Bus System")
    g.fig.suptitle('Impact of SOA Parameters on Model Performance', y=1.03, weight='bold')
    save_path = config.get_evaluation_path('soa_tuning_performance_summary.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MoSOA performance summary plot saved to {save_path}")

def _init_positions(num_agents, dim, upper_bound, lower_bound):
    if isinstance(upper_bound, (int, float)): upper_bound = np.full(dim, upper_bound)
    if isinstance(lower_bound, (int, float)): lower_bound = np.full(dim, lower_bound)
    positions = np.zeros((num_agents, dim))
    for i in range(dim):
        positions[:, i] = np.random.uniform(lower_bound[i], upper_bound[i], num_agents)
    return positions

def MoSoa(num_agents, max_iter, lower_bound, upper_bound, dim, objective_func):
    """Seagull Optimization Algorithm for hyperparameter tuning."""
    pbar = tqdm(range(max_iter), desc=f"MoSOA (S:{num_agents}, I:{max_iter})", leave=False)
    best_position, best_score = np.zeros(dim), float('inf')
    positions = _init_positions(num_agents, dim, upper_bound, lower_bound)
    lambda_beta, beta_max = 5.0, 2.0
    for l in pbar:
        fitness_all = [objective_func(np.clip(p, lower_bound, upper_bound)) for p in positions]
        valid_fitness = [(f, i) for i, f in enumerate(fitness_all) if f is not None and f != float('inf')]
        if valid_fitness:
            current_best_score_iter, best_agent_idx = min(valid_fitness, key=lambda item: item[0])
            if current_best_score_iter < best_score:
                best_score, best_position = current_best_score_iter, positions[best_agent_idx].copy()
        fc, beta = 2 - l * (2 / max_iter), beta_max * np.exp(-lambda_beta * (l / max_iter))
        for i in range(num_agents):
            A = fc * (1 - l / max_iter); C = 2 * A * np.random.rand(dim) - A
            D_beta = abs(C * best_position - positions[i, :])
            positions[i, :] = D_beta * np.exp(np.random.rand() * 2 * np.pi) * np.cos(2 * np.pi * np.random.rand()) + best_position
        pbar.set_description(f"MoSOA (S:{num_agents}, I:{max_iter}) | Best MSE: {best_score:.6f}")
    return best_score, best_position

def main():
    class Args:
        # Running a smaller set for the meta-analysis example
        models_to_test = ['PIGCLSTM', 'ResnetPIGCGRU'] 
        seed = 42
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
    ADAPTIVE_MODELS = ['adaptiveGCN', 'PIGCN', 'ResnetPIGCGRU', 'ResnetPIGCLSTM', 'PIGCLSTM', 'PIGCGRU']
    
    # Define the SOA configurations to test.
    soa_configs_to_test = [
        {'num_seagulls': 5, 'max_iterations': 5},
        {'num_seagulls': 5, 'max_iterations': 10},
        {'num_seagulls': 10, 'max_iterations': 10},
    ]
    all_tuning_results = []
    
    bus_systems_to_test = base_config.NUM_BUSES if isinstance(base_config.NUM_BUSES, list) else [base_config.NUM_BUSES]

    # Primary loop to test different MoSOA configurations.
    for MoSOA_config in MoSOA_configs_to_test:
        num_seagulls, max_iterations = MoSOA_config['num_seagulls'], MoSOA_config['max_iterations']
        print(f"\n{'#'*80}\n# TESTING MoSOA CONFIG: Seagulls={num_seagulls}, Iterations={max_iterations}\n{'#'*80}")

        for num_buses in bus_systems_to_test:
            print(f"\n{'='*50}\n# Bus System: {num_buses}\n{'='*50}")
            case_name = f"case{num_buses}"
            try:
                data_tuple = load_power_system_data(base_config, case_name)
                _features, _adjacency, _ybus_list, _targets, _energy_coeffs, _carbon_coeffs, _sample_map, _normalizer = data_tuple
            except FileNotFoundError as e:
                print(f"[CRITICAL ERROR] {e}"); continue

            for model_name in args.models_to_test:
                print(f"\n--- Starting search for: {model_name} ---")
                model_config = model_config_map[model_name]
                
                is_sequential = 'LSTM' in model_name.upper() or 'GRU' in model_name.upper()
                is_physics_informed = 'PI' in model_name

                param_bounds = {'HIDDEN_DIM': model_config.HIDDEN_DIM_RANGE, 'NUM_GC_LAYERS': model_config.NUM_GC_LAYERS_RANGE}
                if is_physics_informed: param_bounds['LAMBDA_P'] = (1.0, 50.0)
                if is_sequential: param_bounds.update({'SEQUENCE_LENGTH': model_config.SEQUENCE_LENGTH_RANGE, 'RNN_LAYERS': model_config.RNN_LAYERS_RANGE})
                if model_name in ADAPTIVE_MODELS:
                    param_bounds.update({'EMBEDDING_DIM': model_config.EMBEDDING_DIM_RANGE, 'PHI': model_config.PHI_RANGE})
                
                param_keys, dim, lower_bounds, upper_bounds = list(param_bounds.keys()), len(param_bounds), [b[0] for b in param_bounds.values()], [b[1] for b in param_bounds.values()]

                def objective_function(params_array):
                    params = {key: val for key, val in zip(param_keys, params_array)}
                    for k in ['HIDDEN_DIM', 'NUM_GC_LAYERS', 'SEQUENCE_LENGTH', 'RNN_LAYERS', 'EMBEDDING_DIM']:
                        if k in params: params[k] = int(round(params[k]))
                    run_config = copy.deepcopy(base_config)
                    for key, value in params.items(): setattr(run_config, key.upper(), value)
                    run_config.NUM_BUSES = num_buses
                    try:
                        loaders = create_data_loaders(_features, _adjacency, _ybus_list, _targets, _energy_coeffs, _carbon_coeffs, _sample_map, run_config, is_static=(not is_sequential))
                        train_loader, val_loader, _ = loaders
                        model_kwargs = { 'feature_dim': model_config.FEATURE_DIM, 'hidden_dim': params['HIDDEN_DIM'], 'num_gc_layers': params['NUM_GC_LAYERS'], 
                                         'num_buses': num_buses, 'dropout': model_config.DROPOUT }
                        if is_sequential: model_kwargs['rnn_layers'] = params['RNN_LAYERS']
                        if model_name in ADAPTIVE_MODELS:
                            model_kwargs.update({'embedding_dim': params['EMBEDDING_DIM'], 'phi': params['PHI']})
                        
                        model = model_class_map[model_name](**model_kwargs).to(device)
                        criterion = PowerSystemLoss(config=run_config, normalizer=_normalizer).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=run_config.LEARNING_RATE)
                        trainer = PowerSystemTrainer(model, criterion, optimizer, run_config, device)
                        val_mse = trainer.train(train_loader, val_loader)
                        return val_mse
                    except Exception:
                        return float('inf')
                    finally:
                        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

                best_mse_for_config, _ = MoSoa(num_seagulls, max_iterations, lower_bounds, upper_bounds, dim, objective_function)
                
                if best_mse_for_config != float('inf'):
                    all_tuning_results.append({
                        'num_seagulls': num_seagulls,
                        'max_iterations': max_iterations,
                        'num_buses': num_buses,
                        'model_name': model_name,
                        'best_mse': best_mse_for_config
                    })
    
    # After all experiments, analyze and plot the tuning results.
    if not all_tuning_results:
        print("\nNo successful tuning runs were completed. Cannot generate summary.")
    else:
        print(f"\n{'#'*80}\n# MoSOA TUNING ANALYSIS\n{'#'*80}")
        tuning_df = pd.DataFrame(all_tuning_results)
        summary_path = base_config.get_evaluation_path('MoSOA_tuning_summary.csv')
        tuning_df.to_csv(summary_path, index=False)
        print("MoSOA tuning summary saved to:", summary_path)
        print(tuning_df.round(6).to_string())

        plot_MoSOA_performance(tuning_df, base_config)

        # --- START: FINAL FIX ---
        # Note: The renewable impact plots are no longer generated in this version
        # as the focus is on the meta-analysis of the tuning algorithm itself.
        # If you want to generate them for the absolute best model found across all
        # runs, you would need to add logic to track that best model's parameters
        # and checkpoint, then re-train and evaluate it after the main loop.
        # --- END: FINAL FIX ---

if __name__ == '__main__':
    main()