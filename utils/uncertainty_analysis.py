"""
Uncertainty Quantification and Visualization for Power System State Estimation.
Generates spatial and temporal uncertainty visualizations for trained models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import networkx as nx
import pandapower as pp
import pandapower.networks as pn
from typing import Dict, Tuple, List
import torch


def load_network_topology(case_name: str) -> Tuple[pp.pandapowerNet, nx.Graph, Dict]:
    """
    Load network topology and extract bus positions for visualization.
    
    Args:
        case_name: Name of the test case (e.g., "case33", "case57", "case118")
    
    Returns:
        net: Pandapower network object
        G: NetworkX graph
        pos: Dictionary of bus positions {bus_id: (x, y)}
    """
    # Load the appropriate network
    if case_name == "case33":
        net = pn.case33bw()
    elif case_name == "case57":
        net = pn.case57()
    elif case_name == "case118":
        net = pn.case118()
    else:
        raise ValueError(f"Unknown case: {case_name}")
    
    # Create NetworkX graph from pandapower network
    G = nx.Graph()
    
    # Add nodes (buses)
    for bus_idx in net.bus.index:
        G.add_node(bus_idx)
    
    # Add edges (lines)
    for _, line in net.line.iterrows():
        G.add_edge(line.from_bus, line.to_bus)
    
    # Generate positions using spring layout (will look similar to typical power system layouts)
    if case_name == "case33":
        # For case33, use a hierarchical layout (it's a radial feeder)
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    else:
        # For larger systems, use spring layout with more iterations
        pos = nx.spring_layout(G, seed=42, k=1, iterations=100)
    
    return net, G, pos


def calculate_uncertainty_metrics(predictions: np.ndarray, targets: np.ndarray, 
                                  renewable_fractions: np.ndarray) -> Dict:
    """
    Calculate uncertainty metrics for each renewable fraction.
    
    Args:
        predictions: Shape [n_samples, n_buses, n_features]
        targets: Shape [n_samples, n_buses, n_features]
        renewable_fractions: Shape [n_samples] - renewable fraction for each sample
    
    Returns:
        Dictionary containing uncertainty metrics for each renewable fraction
    """
    # Extract voltage magnitude (feature 0)
    v_pred = predictions[:, :, 0]  # [n_samples, n_buses]
    v_true = targets[:, :, 0]
    
    # Calculate errors
    errors = v_pred - v_true  # [n_samples, n_buses]
    
    # Get unique renewable fractions
    # IMPORTANT: Round to 1 decimal place to avoid floating point precision issues
    # This ensures keys like 0.2 don't become 0.19999999 or 0.20000001
    renewable_fractions_rounded = np.round(renewable_fractions, decimals=1)
    unique_fractions = np.unique(renewable_fractions_rounded)
    
    uncertainty_data = {}
    
    for frac in unique_fractions:
        # Get indices for this fraction (using rounded values for stable comparison)
        mask = renewable_fractions_rounded == frac
        frac_errors = errors[mask]  # [n_frac_samples, n_buses]
        
        # Spatial uncertainty: std across time for each bus
        spatial_uncertainty = np.std(frac_errors, axis=0)  # [n_buses]
        
        # Temporal uncertainty: mean absolute error across buses for each timestep
        temporal_uncertainty = np.mean(np.abs(frac_errors), axis=1)  # [n_frac_samples]
        
        # Use rounded float as key to match expected_fractions exactly
        uncertainty_data[round(float(frac), 1)] = {
            'spatial': spatial_uncertainty,
            'temporal': temporal_uncertainty,
            'mean_spatial': np.mean(spatial_uncertainty),
            'max_spatial': np.max(spatial_uncertainty),
            'mean_temporal': np.mean(temporal_uncertainty)
        }
    
    return uncertainty_data


def plot_spatial_comparison_grid(uncertainty_data: Dict, case_name: str, 
                                 output_path: str, model_name: str = ""):
    """
    Generate 6-panel spatial uncertainty comparison grid.
    
    Args:
        uncertainty_data: Dictionary with uncertainty metrics per renewable fraction
        case_name: Name of the test case
        output_path: Where to save the output image
        model_name: Optional model name for title
    """
    # Load network topology
    net, G, pos = load_network_topology(case_name)
    
    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # FIXED: Expect all 6 standard renewable fractions
    expected_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    available_fractions = set(uncertainty_data.keys())
    
    # Color map limits (use global min/max for consistent color scale across available data)
    if available_fractions:
        all_spatial = [uncertainty_data[f]['spatial'] for f in available_fractions]
        vmin = min(s.min() for s in all_spatial)
        vmax = max(s.max() for s in all_spatial)
    else:
        vmin, vmax = 0, 0.001
    
    for idx, frac in enumerate(expected_fractions):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        if frac in uncertainty_data:
            # Data available - plot it
            spatial_unc = uncertainty_data[frac]['spatial']
            
            # Draw network
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=2, edge_color='gray')
            
            # Draw nodes colored by uncertainty
            nodes = nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_color=spatial_unc,
                node_size=500,
                cmap='YlOrRd',
                vmin=vmin,
                vmax=vmax,
                edgecolors='black',
                linewidths=1.5
            )
            
            # Add node labels
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
            
            # Title
            ax.set_title(f'{int(frac*100)}% Renewables\n(Mean σ: {uncertainty_data[frac]["mean_spatial"]:.4f} p.u.)',
                        fontsize=12, fontweight='bold')
        else:
            # Data missing - show placeholder (shouldn't happen with stratified split)
            ax.text(0.5, 0.5, f'{int(frac*100)}% Renewables\n(No data available)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
            ax.set_title(f'{int(frac*100)}% Renewables', fontsize=12, fontweight='bold')
            print(f"[Uncertainty] WARNING: No data for {int(frac*100)}% renewables - check data generation")
        
        ax.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.get_axes(), orientation='horizontal', 
                       pad=0.05, aspect=40, shrink=0.8)
    cbar.set_label('Uncertainty σ (p.u.)', fontsize=14, fontweight='bold')
    
    # Overall title
    title = f'Spatial Uncertainty Map - {case_name.upper()}'
    if model_name:
        title += f' - {model_name}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    # Consolidated output - printed once at the end


def plot_temporal_comparison_curves(uncertainty_data: Dict, case_name: str,
                                   output_path: str, model_name: str = "", config=None):
    """
    Generate temporal uncertainty comparison with curves for available renewable fractions.
    
    Args:
        uncertainty_data: Dictionary with uncertainty metrics per renewable fraction
        case_name: Name of the test case
        output_path: Where to save the output image
        model_name: Optional model name for title
        config: Optional config object to check if using time-series mode
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # FIXED: Use only available fractions, handle missing gracefully
    fractions = sorted(uncertainty_data.keys())
    
    if not fractions:
        # No data available at all
        ax.text(0.5, 0.5, 'No uncertainty data available in test set', 
               transform=ax.transAxes, ha='center', va='center',
               fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean System Uncertainty σ_t (p.u.)', fontsize=14, fontweight='bold')
        ax.set_title(f'Temporal Uncertainty - {case_name.upper()} - {model_name}', 
                    fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Uncertainty] WARNING: No test data for temporal comparison: {output_path}")
        return
    
    # Continue with available fractions
    expected_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    missing_fractions = [f for f in expected_fractions if f not in fractions]
    
    # Warn about missing fractions (shouldn't happen with stratified split, but check anyway)
    if missing_fractions:
        missing_pct = [int(f*100) for f in missing_fractions]
        print(f"[Uncertainty] INFO: Test set missing renewable fractions: {missing_pct}% (reduced data or edge case)")
    
    # Color map for different renewable fractions
    colors = plt.cm.viridis(np.linspace(0, 1, len(fractions)))
    
    # Determine if we should use hours or timesteps for x-axis
    use_time_series = getattr(config, 'USE_TIME_SERIES', False) if config else False
    hours_per_day = getattr(config, 'HOURS_PER_DAY', 24) if config else 24
    
    for frac, color in zip(fractions, colors):
        temporal_unc = uncertainty_data[frac]['temporal']
        n_points = len(temporal_unc)
        
        if use_time_series:
            # TIME-SERIES MODE: Map timesteps to hours of day (modulo 24)
            # This shows the daily cycle pattern regardless of number of samples
            x_values = np.arange(n_points) % hours_per_day
            # Sort by hour of day for cleaner visualization
            sort_idx = np.argsort(x_values)
            x_values_sorted = x_values[sort_idx]
            temporal_unc_sorted = temporal_unc[sort_idx]
            
            ax.plot(x_values_sorted, temporal_unc_sorted, 
                   label=f'{int(frac*100)}% Renewables (μ={uncertainty_data[frac]["mean_temporal"]:.4f})',
                   color=color, linewidth=2, alpha=0.8, marker='o', markersize=3)
        else:
            # MONTE CARLO MODE: X-axis shows timesteps
            x_values = np.arange(n_points)
            ax.plot(x_values, temporal_unc, 
                   label=f'{int(frac*100)}% Renewables (μ={uncertainty_data[frac]["mean_temporal"]:.4f})',
                   color=color, linewidth=2, alpha=0.8, marker='o', markersize=4)
    
    # Labels and title
    if use_time_series:
        ax.set_xlabel('Hour of Day', fontsize=14, fontweight='bold')
        # Set x-axis to show 0-23 hours
        ax.set_xlim(-0.5, hours_per_day - 0.5)
        ax.set_xticks(np.arange(0, hours_per_day, 3))  # Show every 3 hours
        ax.set_xticklabels([f'{h}:00' for h in range(0, hours_per_day, 3)])  # Format as times
    else:
        ax.set_xlabel('Timestep', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Mean System Uncertainty σ_t (p.u.)', fontsize=14, fontweight='bold')
    
    title = f'Temporal Uncertainty Curve - {case_name.upper()}'
    if model_name:
        title += f' - {model_name}'
    if use_time_series:
        title += ' (Daily Cycle Pattern)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    # Consolidated output - printed once at the end


def generate_uncertainty_visualizations(predictions: np.ndarray, targets: np.ndarray,
                                       renewable_fractions: np.ndarray, case_name: str,
                                       output_dir: str, model_name: str = "", config=None):
    """
    Main function to generate all uncertainty visualizations.
    
    Args:
        predictions: Model predictions [n_samples, n_buses, n_features]
        targets: True values [n_samples, n_buses, n_features]
        renewable_fractions: Renewable fraction for each sample [n_samples]
        case_name: Test case name (e.g., "case33")
        output_dir: Directory to save outputs
        model_name: Optional model name for titles and filenames
        config: Optional config object for time-series mode detection
    
    Returns:
        uncertainty_data: Dictionary with all calculated metrics
    """
    # Calculate uncertainty metrics (silent - will print summary at end)
    uncertainty_data = calculate_uncertainty_metrics(predictions, targets, renewable_fractions)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate spatial comparison grid
    spatial_output = os.path.join(output_dir, 'uncertainty_spatial.png')
    plot_spatial_comparison_grid(uncertainty_data, case_name, spatial_output, model_name)
    
    # Generate temporal comparison curves (hours if time-series mode)
    temporal_output = os.path.join(output_dir, 'uncertainty_temporal.png')
    plot_temporal_comparison_curves(uncertainty_data, case_name, temporal_output, model_name, config)
    
    # Print consolidated message
    print(f"[Uncertainty] Saved all plots for {model_name}")
    
    return uncertainty_data


def split_by_renewable_fraction(predictions: torch.Tensor, targets: torch.Tensor,
                                renewable_fractions: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to convert tensors to numpy and prepare for uncertainty analysis.
    
    Args:
        predictions: Tensor [n_samples, n_buses, n_features]
        targets: Tensor [n_samples, n_buses, n_features]
        renewable_fractions: Tensor [n_samples]
    
    Returns:
        predictions_np, targets_np, renewable_fractions_np as numpy arrays
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(renewable_fractions, torch.Tensor):
        renewable_fractions = renewable_fractions.cpu().numpy()
    
    return predictions, targets, renewable_fractions
