"""
Robustness Evaluation: Integrated Contingency Analysis

Evaluates model performance under N-1 contingencies and generates comparison plots.
Integrates contingency_analysis.py into the main evaluation pipeline.
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')

from utils.evaluation import evaluate_model
from utils.metrics import compute_metrics


def evaluate_model_robustness(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
                              device: torch.device, config: Any, normalizer: Any,
                              case_name: str, output_dir: str, model_name: str = "",
                              top_k_contingencies: int = 10, contingency_method: str = 'power_flow'):
    """
    Evaluate model robustness under N-1 contingencies.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        config: Configuration object
        normalizer: Data normalizer
        case_name: Name of the test case (e.g., "case33")
        output_dir: Directory to save plots
        model_name: Optional model name for title
        top_k_contingencies: Number of critical contingencies to test
        contingency_method: Method to identify critical lines ('power_flow', 'centrality', 'historical')
    
    Returns:
        Dictionary with baseline and contingency performance metrics
    """
    # Step 1: Evaluate baseline performance (normal conditions)
    baseline_metrics = evaluate_model(model, test_loader, device, config, normalizer, is_sequential=False)
    
    baseline_mse = baseline_metrics.get('mse', float('inf'))
    baseline_power_violation = baseline_metrics.get('power_violation', float('inf'))
    baseline_voltage_violation = baseline_metrics.get('voltage_violation', float('inf'))
    
    # Step 2: Identify critical lines using ContingencyAnalyzer
    # Note: This requires loading the pandapower network
    # For now, we'll use a simplified approach: test top K lines by index
    # In a full implementation, you would load the network and use ContingencyAnalyzer
    try:
        # Try to load network for proper contingency analysis
        import pandapower.networks as pn
        
        if case_name.lower() == 'case33':
            net = pn.case33bw()
        elif case_name.lower() == 'case57':
            net = pn.case57()
        elif case_name.lower() == 'case118':
            net = pn.case118()
        else:
            critical_lines = list(range(min(top_k_contingencies, 50)))  # Fallback
            net = None
    except Exception as e:
        critical_lines = list(range(min(top_k_contingencies, 50)))
        net = None
    
    if net is not None:
        # Use ContingencyAnalyzer to identify critical lines
        analyzer = ContingencyAnalyzer(net)
        
        if contingency_method == 'power_flow':
            critical_lines = analyzer.identify_critical_lines_by_power_flow(net, top_k=top_k_contingencies)
        elif contingency_method == 'centrality':
            critical_lines = analyzer.identify_critical_lines_by_centrality(net, top_k=top_k_contingencies)
        else:
            # Fallback to power flow
            critical_lines = analyzer.identify_critical_lines_by_power_flow(net, top_k=top_k_contingencies)
    
    # Step 3: Evaluate model under each contingency
    # Real implementation: Modify Ybus and adjacency matrices, then re-evaluate
    from utils.contingency_ybus import create_contingency_ybus_batch, create_contingency_adjacency_batch
    from utils.contingency_analysis import ContingencyAnalyzer
    
    contingency_results = []
    
    # Initialize analyzer for islanding checks
    analyzer = None
    if net is not None:
        analyzer = ContingencyAnalyzer(net)
    
    for i, line_idx in enumerate(critical_lines[:top_k_contingencies]):
        if net is None:
            # Fallback: skip if network not available
            continue
        
        # Check if contingency causes islanding
        contingency_test = analyzer.test_contingency(net, line_idx, run_power_flow=False)
        if contingency_test.get('islanding', False):
            # Skip islanding cases (system is disconnected)
            contingency_results.append({
                'line_idx': line_idx,
                'mse': float('inf'),
                'power_violation': float('inf'),
                'voltage_violation': float('inf'),
                'islanding': True,
                'error': 'Islanding detected'
            })
            continue
        
        # Create a modified test loader with contingency Ybus and adjacency
        # We'll iterate through the test loader and modify Ybus/adjacency on-the-fly
        model.eval()
        contingency_predictions = []
        contingency_targets = []
        contingency_ybus_list = []
        contingency_features_list = []
        contingency_bus_types_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Get original data
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)
                ybus_original = batch['ybus_matrix'].to(device)
                adjacency_original = batch['adjacency'].to(device)
                bus_types = batch.get('bus_types', None)
                if bus_types is not None:
                    bus_types = bus_types.to(device)
                
                # Modify Ybus and adjacency for this contingency
                ybus_contingency = create_contingency_ybus_batch(ybus_original, net, line_idx)
                adjacency_contingency = create_contingency_adjacency_batch(adjacency_original, net, line_idx)
                
                # Get model predictions with contingency Ybus/adjacency
                # Note: Models use adjacency, not Ybus directly, but we pass modified Ybus for physics loss
                try:
                    if features.dim() == 4:  # Sequential model
                        outputs = model(features, adjacency_contingency, bus_types=bus_types)
                    else:  # Static model
                        outputs = model(features, adjacency_contingency, bus_types=bus_types)
                except TypeError:
                    # Model doesn't support bus_types
                    if features.dim() == 4:
                        outputs = model(features, adjacency_contingency)
                    else:
                        outputs = model(features, adjacency_contingency)
                
                # Store for metric computation
                contingency_predictions.append(outputs.cpu())
                contingency_targets.append(targets.cpu())
                contingency_ybus_list.append(ybus_contingency.cpu())
                contingency_features_list.append(features.cpu())
                if bus_types is not None:
                    contingency_bus_types_list.append(bus_types.cpu())
        
        # Concatenate all batches
        all_predictions = torch.cat(contingency_predictions, dim=0)
        all_targets = torch.cat(contingency_targets, dim=0)
        all_ybus = torch.cat(contingency_ybus_list, dim=0)
        all_features = torch.cat(contingency_features_list, dim=0)
        all_bus_types = torch.cat(contingency_bus_types_list, dim=0) if contingency_bus_types_list else None
        
        # Denormalize predictions and targets
        predictions_denorm = normalizer.denormalize(all_predictions)
        targets_denorm = normalizer.denormalize(all_targets)
        
        # Compute metrics using modified Ybus and measurements
        # FIXED: Now passes measurements to compute actual physics violations
        from utils.metrics import compute_metrics
        
        # Extract measurements from features (for sequential models, use last timestep)
        if all_features.dim() == 4:  # Sequential: [batch, seq_len, buses, features]
            measurements_for_metrics = all_features[:, -1, :, :]  # Use last timestep
        else:  # Static: [batch, buses, features]
            measurements_for_metrics = all_features
        
        contingency_metrics = compute_metrics(
            predictions_denorm, targets_denorm, all_ybus, config, 
            bus_types=all_bus_types, measurements=measurements_for_metrics
        )
        
        contingency_results.append({
            'line_idx': line_idx,
            'mse': contingency_metrics.get('mse', float('inf')),
            'power_violation': contingency_metrics.get('power_violation', float('inf')),
            'voltage_violation': contingency_metrics.get('voltage_violation', float('inf')),
            'islanding': False
        })
    
    # Step 4: Generate comparison bar chart
    plot_contingency_comparison(baseline_metrics, contingency_results, case_name, 
                               output_dir, model_name)
    
    return {
        'baseline': baseline_metrics,
        'contingencies': contingency_results
    }


def plot_contingency_comparison(baseline_metrics: Dict[str, float],
                                contingency_results: List[Dict[str, Any]],
                                case_name: str, output_dir: str, model_name: str = ""):
    """
    Generate bar chart comparing baseline vs contingency performance.
    
    Args:
        baseline_metrics: Dictionary with baseline performance metrics
        contingency_results: List of dictionaries with contingency performance metrics
        case_name: Name of the test case
        output_dir: Directory to save plot
        model_name: Optional model name for title
    """
    if not contingency_results:
        return
    
    # Extract metrics
    baseline_mse = baseline_metrics.get('mse', 0.0)
    baseline_power_vio = baseline_metrics.get('power_violation', 0.0)
    baseline_voltage_vio = baseline_metrics.get('voltage_violation', 0.0)
    
    # Filter out islanding cases for plotting (they have inf values)
    plot_results = [r for r in contingency_results if not r.get('islanding', False)]
    islanding_results = [r for r in contingency_results if r.get('islanding', False)]
    
    contingency_mse = [r['mse'] for r in plot_results]
    contingency_power_vio = [r['power_violation'] for r in plot_results]
    contingency_voltage_vio = [r['voltage_violation'] for r in plot_results]
    line_indices = [r['line_idx'] for r in plot_results]
    islanding_line_indices = [r['line_idx'] for r in islanding_results]
    
    # Create 1x3 subplot: MSE, Power Violation, Voltage Violation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title = f'Robustness Analysis: Baseline vs N-1 Contingencies - {case_name.upper()}'
    if model_name:
        title += f' - {model_name}'
    if islanding_results:
        title += f' ({len(islanding_results)} islanding cases excluded)'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(plot_results) + 1)  # +1 for baseline
    width = 0.6
    
    # Plot 1: MSE
    ax = axes[0]
    mse_values = [baseline_mse] + contingency_mse
    labels = ['Baseline'] + [f'Line {idx}' for idx in line_indices]
    colors = ['green'] + ['orange'] * len(plot_results)
    bars = ax.bar(x_pos, mse_values, width, color=colors)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Prediction Error (MSE)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    if max(mse_values) > 0:
        ax.set_yscale('log')
    if islanding_results:
        ax.text(0.02, 0.98, f'Islanding: Lines {islanding_line_indices}', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Plot 2: Power Violation
    ax = axes[1]
    power_vio_values = [baseline_power_vio] + contingency_power_vio
    colors = ['green'] + ['red'] * len(plot_results)
    bars = ax.bar(x_pos, power_vio_values, width, color=colors)
    ax.set_ylabel('Power Violation (p.u.)', fontsize=12)
    ax.set_title('Power Balance Violation', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    if max(power_vio_values) > 0:
        ax.set_yscale('log')
    
    # Plot 3: Voltage Violation
    ax = axes[2]
    voltage_vio_values = [baseline_voltage_vio] + contingency_voltage_vio
    colors = ['green'] + ['purple'] * len(plot_results)
    bars = ax.bar(x_pos, voltage_vio_values, width, color=colors)
    ax.set_ylabel('Voltage Violation (p.u.)', fontsize=12)
    ax.set_title('Voltage Limit Violation', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    if max(voltage_vio_values) > 0:
        ax.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'robustness_contingency_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

