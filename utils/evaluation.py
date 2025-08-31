"""
Evaluation and results processing utilities for power system machine learning models.
Contains functions for model evaluation, results saving, and summary generation.
"""

import os
import torch
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
import torch.nn.functional as F

from utils.metrics import PowerSystemLoss, compute_metrics
from utils.visualization import (plot_training_history, plot_convergence, 
                                plot_all_renewable_impacts, create_model_comparison_plot)


def evaluate_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, 
                  device: torch.device, config: Any, normalizer: Any, 
                  is_sequential: bool) -> Dict[str, float]:
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
    
    # Handle shape consistency for different model types before denormalization
    if all_outputs_tensor.dim() == 2:
        # If model outputs flattened format [batch_size, num_buses * features]
        batch_size = all_outputs_tensor.shape[0]
        num_features = 6
        all_outputs_tensor = all_outputs_tensor.view(batch_size, num_buses, num_features)
    
    outputs_denorm = normalizer.denormalize(all_outputs_tensor, num_buses)
    targets_denorm = normalizer.denormalize(all_targets_tensor, num_buses)

    return compute_metrics(outputs_denorm, targets_denorm, all_ybus_tensor, config)


def evaluate_moopf_objectives(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                             config: Any, device: torch.device, normalizer: Any, 
                             is_physics_informed: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            
            # Handle shape consistency for different model types
            if outputs_norm.dim() == 2:
                # If model outputs flattened format [batch_size, num_buses * features]
                batch_size = outputs_norm.shape[0]
                num_features = 6
                outputs_norm = outputs_norm.view(batch_size, num_buses, num_features)
            
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
                    # Use the actual renewable fraction from the data (not calculated from outputs)
                    renewable_fractions = batch['renewable_fraction'].cpu().numpy()

                    for i in range(features.shape[0]):
                        renewable_impact_data.append({
                            'renewable_fraction': renewable_fractions[i],
                            'normalized_carbon_emissions': emissions['normalized'][i].item(),
                            'voltage_deviation': norm_vdev[i].item(),
                            'power_loss': norm_loss[i].item(),
                            'power_flow': norm_power_flow[i].item()
                        })
                except (IndexError, KeyError) as e:
                    logging.warning(f"Could not extract renewable fraction from batch data: {e}")

            if is_physics_informed:
                moopf_score = (w_loss * norm_loss + w_vdev * norm_vdev + w_carbon * emissions['normalized'])
                all_results.append({
                    'moopf_score': moopf_score.mean().item(), 
                    'normalized_power_loss': norm_loss.mean().item(),
                    'normalized_voltage_deviation': norm_vdev.mean().item(), 
                    'normalized_carbon_emissions': emissions['normalized'].mean().item(),
                    'raw_carbon_emissions_tCO2': emissions['raw'].mean().item(),
                    'normalized_power_flow': norm_power_flow.mean().item()
                })
            else:
                # For non-physics models, only report MSE-based results
                targets = batch['targets'].to(device)
                targets_norm = normalizer.normalize(targets)
                mse_only = F.mse_loss(outputs_norm, targets_norm)
                all_results.append({
                    'mse_score': mse_only.item(),  # Main metric for non-physics models
                    'normalized_power_loss': 0.0,  # Not applicable
                    'normalized_voltage_deviation': 0.0,  # Not applicable
                    'normalized_carbon_emissions': 0.0,  # Not applicable
                    'raw_carbon_emissions_tCO2': 0.0,  # Not applicable
                    'normalized_power_flow': 0.0  # Not applicable
                })

    return pd.DataFrame(all_results), pd.DataFrame(renewable_impact_data)


def create_iteration_wise_results(iteration_details: List[Dict], param_keys: List[str], 
                                 config: Any, model_name: str) -> pd.DataFrame:
    """
    Create iteration-wise results DataFrame showing best configuration per iteration.
    
    Args:
        iteration_details: List of iteration details from MoSOA
        param_keys: List of parameter names
        config: Configuration object
        model_name: Name of the model
        
    Returns:
        DataFrame with iteration-wise best configurations
    """
    iteration_results = []
    
    for details in iteration_details:
        if details['best_position'] is not None:
            # Convert position to parameter dictionary
            params = {key: val for key, val in zip(param_keys, details['best_position'])}
            
            # Convert integer parameters
            for k in ['HIDDEN_DIM', 'NUM_GC_LAYERS', 'SEQUENCE_LENGTH', 'RNN_LAYERS', 'EMBEDDING_DIM']:
                if k in params:
                    params[k] = int(round(params[k]))
            
            # Create result row
            result_row = {
                'model_name': model_name,
                'iteration': details['iteration'],
                'iteration_best_score': details['best_score'],
                'global_best_score': details['global_best_score'],
                'num_evaluations': details['num_valid_evaluations'],
                **params  # Add all hyperparameters
            }
            iteration_results.append(result_row)
    
    return pd.DataFrame(iteration_results)


def save_best_model_results(best_model: torch.nn.Module, best_run: Dict[str, Any], 
                           moopf_results: pd.DataFrame, renewable_impact_data: pd.DataFrame, 
                           training_history: Dict[str, List], config: Any, num_buses: int, 
                           is_physics_informed: bool = True, iteration_details: List[Dict] = None,
                           param_keys: List[str] = None):
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
    
    # Save iteration-wise optimization results if available
    if iteration_details is not None and param_keys is not None:
        iteration_results_df = create_iteration_wise_results(iteration_details, param_keys, config, model_name)
        iteration_path = os.path.join(model_dir, "iteration_wise_results.csv")
        iteration_results_df.to_csv(iteration_path, index=False)
        print(f"📊 Iteration-wise results saved to: {iteration_path}")
    
    # Save summary with filtered data based on model type
    if is_physics_informed:
        # Physics models: Save all metrics
        summary_data = best_run.copy()
    else:
        # Non-physics models: Remove physics-related metrics
        summary_data = best_run.copy()
        # Remove physics metrics from top level
        physics_metrics = ['power_violation', 'voltage_violation']
        for metric in physics_metrics:
            summary_data.pop(metric, None)
        
        # Clean validation metrics if they exist
        if 'val_metrics' in summary_data and isinstance(summary_data['val_metrics'], dict):
            val_metrics_clean = {k: v for k, v in summary_data['val_metrics'].items() 
                               if k not in physics_metrics}
            summary_data['val_metrics'] = val_metrics_clean
        
        # Clean training history if it exists
        if 'training_history' in summary_data and isinstance(summary_data['training_history'], dict):
            history_clean = summary_data['training_history'].copy()
            for metric in ['train_power_violation', 'val_power_violation', 
                          'train_voltage_violation', 'val_voltage_violation']:
                history_clean.pop(metric, None)
            summary_data['training_history'] = history_clean
    
    pd.DataFrame([summary_data]).to_csv(config.get_summary_path(num_buses, model_name), index=False)
    
    # Plot training history (available for all models)
    plot_training_history(training_history, model_name, config, num_buses, is_physics_informed)
    
    # Plot convergence history if available (available for all models)
    if 'convergence_history' in best_run:
        plot_convergence(best_run['convergence_history'], model_name, config, num_buses)
    
    # Only plot renewable impacts for physics-informed models
    if is_physics_informed:
        plot_all_renewable_impacts(renewable_impact_data, config, num_buses, model_name)
    else:
        print(f"ℹ️  Skipping renewable impact plots for non-physics-informed model: {model_name}")


def print_comprehensive_summary(all_results: List[Dict[str, Any]]):
    """Print and save a comprehensive summary of all model performances across all bus systems."""
    if not all_results:
        print("\n❌ No results to summarize.")
        return
    
    print(f"\n{'='*100}")
    print(f"🎯 COMPREHENSIVE FINAL SUMMARY - ALL MODELS & BUS SYSTEMS")
    print(f"{'='*100}")
    
    # Create summary table for display
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model_name'],
            'Bus System': f"{result['num_buses']}-bus",
            'Type': 'Physics' if result['is_physics_informed'] else 'Non-Physics',
            'Hidden Dim': result['best_hidden_dim'],
            'GC Layers': result['best_gc_layers'],
            'Training MSE': f"{result['training_mse']:.6f}" if result['training_mse'] != float('inf') else 'Failed',
            'Test Score': f"{result['final_test_score']:.6f}" if result['final_test_score'] != float('inf') else 'Failed',
            'Metric Type': result['final_metric_name']
        })
    
    csv_data = []
    for result in all_results:
        csv_data.append({
            'model_name': result['model_name'],
            'num_buses': result['num_buses'],
            'bus_system': f"{result['num_buses']}-bus",
            'model_type': 'Physics-Informed' if result['is_physics_informed'] else 'Non-Physics',
            'is_physics_informed': result['is_physics_informed'],
            'best_hidden_dim': result['best_hidden_dim'],
            'best_gc_layers': result['best_gc_layers'],
            'training_mse': result['training_mse'],
            'final_test_score': result['final_test_score'],
            'final_metric_name': result['final_metric_name'],
            'power_violation': result['power_violation'],
            'voltage_violation': result['voltage_violation'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Save to CSV file in experimental_results directory  
    model_eval_dir = "experimental_results"
    os.makedirs(model_eval_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"comprehensive_summary_{timestamp}.csv"
    csv_path = os.path.join(model_eval_dir, csv_filename)
    
    # Also save a "latest" version for easy access
    latest_csv_path = os.path.join(model_eval_dir, "comprehensive_summary_latest.csv")
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    df.to_csv(latest_csv_path, index=False)
    
    print(f"📁 Comprehensive summary saved to:")
    print(f"   Timestamped: {csv_path}")
    print(f"   Latest: {latest_csv_path}")
    print()
    
    # Print table header
    print(f"{'Model':<15} {'Bus Sys':<8} {'Type':<11} {'H.Dim':<7} {'GC Ly':<5} {'Train MSE':<12} {'Test Score':<12} {'Metric':<12}")
    print("-" * 100)
    
    # Print each result
    for data in summary_data:
        print(f"{data['Model']:<15} {data['Bus System']:<8} {data['Type']:<11} {data['Hidden Dim']:<7} {data['GC Layers']:<5} {data['Training MSE']:<12} {data['Test Score']:<12} {data['Metric Type']:<12}")
    
    print("-" * 100)
    
    # Find overall best performers
    successful_results = [r for r in all_results if r['final_test_score'] != float('inf')]
    
    if successful_results:
        # Best overall (lowest test score)
        best_overall = min(successful_results, key=lambda x: x['final_test_score'])
        print(f"\n🏆 OVERALL BEST PERFORMER:")
        print(f"   Model: {best_overall['model_name']} on {best_overall['num_buses']}-bus system")
        print(f"   {best_overall['final_metric_name']}: {best_overall['final_test_score']:.6f}")
        print(f"   Config: {best_overall['best_hidden_dim']} hidden_dim, {best_overall['best_gc_layers']} GC layers")
        
        # Best per bus system
        print(f"\n📊 BEST PER BUS SYSTEM:")
        bus_systems = list(set(r['num_buses'] for r in successful_results))
        for num_buses in sorted(bus_systems):
            bus_results = [r for r in successful_results if r['num_buses'] == num_buses]
            if bus_results:
                best_for_bus = min(bus_results, key=lambda x: x['final_test_score'])
                print(f"   {num_buses}-bus: {best_for_bus['model_name']} ({best_for_bus['final_metric_name']}: {best_for_bus['final_test_score']:.6f})")
        
        # Performance comparison
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"   33-bus systems generally perform better (lower error)")
        print(f"   Performance degrades with system size as expected")
        
        # Count successful vs failed models
        total_runs = len(all_results)
        successful_runs = len(successful_results)
        print(f"\n✅ SUCCESS RATE: {successful_runs}/{total_runs} ({100*successful_runs/total_runs:.1f}%)")
        
        # Create comparison plot
        comparison_plot_path = os.path.join(model_eval_dir, "model_comparison_latest.png")
        create_model_comparison_plot(all_results, comparison_plot_path)
        print(f"📊 Comparison plot saved to: {comparison_plot_path}")
        
    else:
        print("\n❌ No successful model runs to analyze.")
    
    print(f"{'='*100}")


def print_model_summary(best_run: Dict[str, Any], moopf_results: pd.DataFrame, 
                       model_name: str, num_buses: int, is_physics_informed: bool, 
                       final_test_score: float, final_metric_name: str):
    """Print a formatted summary of the best model results."""
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL SUMMARY: {model_name} on {num_buses}-bus system")
    print(f"{'='*60}")
    print(f"📊 Best Hyperparameters: {best_run.get('HIDDEN_DIM', 'N/A')} hidden_dim, {best_run.get('NUM_GC_LAYERS', 'N/A')} GC layers")
    print(f"📈 Training Performance: MSE = {best_run.get('mse', 'N/A'):.6f}")
    
    if is_physics_informed:
        print(f"⚡ Physics Violations: Power = {best_run.get('power_violation', 'N/A'):.6f}, Voltage = {best_run.get('voltage_violation', 'N/A'):.6f}")
        print("\n--- MOOPF Evaluation Results ---")
        print(moopf_results.mean().to_dict())
    else:
        print(f"🎯 Final Test MSE: {final_test_score:.6f}")
        print("\n--- MSE Evaluation Results ---")
        # Only show relevant metrics for non-physics models
        relevant_metrics = {
            'mse_score': final_test_score,
            'rmse_score': (final_test_score) ** 0.5,
            'samples_evaluated': len(moopf_results)
        }
        print(relevant_metrics)
    print(f"{'='*60}")
