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
    unique_fractions = np.unique(renewable_fractions)
    
    uncertainty_data = {}
    
    for frac in unique_fractions:
        # Get indices for this fraction
        mask = renewable_fractions == frac
        frac_errors = errors[mask]  # [n_frac_samples, n_buses]
        
        # Spatial uncertainty: std across time for each bus
        spatial_uncertainty = np.std(frac_errors, axis=0)  # [n_buses]
        
        # Temporal uncertainty: mean absolute error across buses for each timestep
        temporal_uncertainty = np.mean(np.abs(frac_errors), axis=1)  # [n_frac_samples]
        
        uncertainty_data[float(frac)] = {
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
    
    # Sort renewable fractions
    fractions = sorted(uncertainty_data.keys())
    
    # Color map limits (use global min/max for consistent color scale)
    all_spatial = [uncertainty_data[f]['spatial'] for f in fractions]
    vmin = min(s.min() for s in all_spatial)
    vmax = max(s.max() for s in all_spatial)
    
    for idx, frac in enumerate(fractions):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
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
    print(f"[Uncertainty] Saved spatial comparison: {output_path}")


def plot_temporal_comparison_curves(uncertainty_data: Dict, case_name: str,
                                   output_path: str, model_name: str = ""):
    """
    Generate temporal uncertainty comparison with 6 curves overlaid.
    
    Args:
        uncertainty_data: Dictionary with uncertainty metrics per renewable fraction
        case_name: Name of the test case
        output_path: Where to save the output image
        model_name: Optional model name for title
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort renewable fractions
    fractions = sorted(uncertainty_data.keys())
    
    # Color map for different renewable fractions
    colors = plt.cm.viridis(np.linspace(0, 1, len(fractions)))
    
    for frac, color in zip(fractions, colors):
        temporal_unc = uncertainty_data[frac]['temporal']
        timesteps = np.arange(len(temporal_unc))
        
        # Plot with label
        ax.plot(timesteps, temporal_unc, 
               label=f'{int(frac*100)}% Renewables (μ={uncertainty_data[frac]["mean_temporal"]:.4f})',
               color=color, linewidth=2, alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Timestep (e.g., hour)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean System Uncertainty σ_t (p.u.)', fontsize=14, fontweight='bold')
    
    title = f'Temporal Uncertainty Curve - {case_name.upper()}'
    if model_name:
        title += f' - {model_name}'
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


def generate_uncertainty_visualizations(predictions: np.ndarray, targets: np.ndarray,
                                       renewable_fractions: np.ndarray, case_name: str,
                                       output_dir: str, model_name: str = ""):
    """
    Main function to generate all uncertainty visualizations.
    
    Args:
        predictions: Model predictions [n_samples, n_buses, n_features]
        targets: True values [n_samples, n_buses, n_features]
        renewable_fractions: Renewable fraction for each sample [n_samples]
        case_name: Test case name (e.g., "case33")
        output_dir: Directory to save outputs
        model_name: Optional model name for titles and filenames
    
    Returns:
        uncertainty_data: Dictionary with all calculated metrics
    """
    print(f"\n[Uncertainty] Generating uncertainty visualizations for {case_name} - {model_name}...")
    
    # Calculate uncertainty metrics
    uncertainty_data = calculate_uncertainty_metrics(predictions, targets, renewable_fractions)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate spatial comparison grid
    spatial_output = os.path.join(output_dir, 'uncertainty_spatial_comparison.png')
    plot_spatial_comparison_grid(uncertainty_data, case_name, spatial_output, model_name)
    
    # Generate temporal comparison curves
    temporal_output = os.path.join(output_dir, 'uncertainty_temporal_comparison.png')
    plot_temporal_comparison_curves(uncertainty_data, case_name, temporal_output, model_name)
    
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

