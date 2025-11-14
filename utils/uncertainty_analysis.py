"""
Uncertainty Quantification and Visualization for Power System OPF.
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


def calculate_predicted_uncertainty_metrics(model_outputs: np.ndarray, renewable_fractions: np.ndarray,
                                            min_sigma: float = 0.01, max_sigma: float = 10.0) -> Dict:
    """
    Calculate PREDICTED uncertainty metrics from model outputs (heteroscedastic mode).
    
    This extracts the model's predicted uncertainties (log_sigma values) and analyzes them
    spatially and temporally. This is different from calculate_uncertainty_metrics which
    uses actual error statistics.
    
    Args:
        model_outputs: Shape [n_samples, n_buses, 4] - Model outputs in heteroscedastic mode
                      [var1_pred, var2_pred, log_sigma_var1, log_sigma_var2]
        renewable_fractions: Shape [n_samples] - renewable fraction for each sample
        min_sigma: Minimum sigma value for clamping
        max_sigma: Maximum sigma value for clamping
    
    Returns:
        Dictionary containing predicted uncertainty metrics for each renewable fraction
    """
    # Extract predicted uncertainties
    log_sigma_var1 = model_outputs[:, :, 2]  # [n_samples, n_buses]
    log_sigma_var2 = model_outputs[:, :, 3]  # [n_samples, n_buses]
    
    # Convert to sigma (standard deviation) using natural parametrization
    # σ² = -1/(2η2), so σ = sqrt(-1/(2η2))
    # Since log_sigma = 0.5 * log(σ²), we have: σ = exp(log_sigma)
    sigma_var1 = np.exp(log_sigma_var1)
    sigma_var2 = np.exp(log_sigma_var2)
    
    # NO CLAMPING: Paper (Immer et al., NeurIPS 2023) does not clamp sigma values
    # Clamping can bias results and is not theoretically justified
    # If sigma values are extreme, it indicates model calibration issues that should be addressed
    
    # Combined uncertainty: Use RMS (root mean square) for proper uncertainty combination
    # RMS preserves units and is standard for combining uncertainties
    # Alternative: Report var1 and var2 separately (more informative)
    sigma_combined = np.sqrt((sigma_var1**2 + sigma_var2**2) / 2.0)  # [n_samples, n_buses] - RMS
    
    # Get unique renewable fractions
    renewable_fractions_rounded = np.round(renewable_fractions, decimals=1)
    unique_fractions = np.unique(renewable_fractions_rounded)
    
    uncertainty_data = {}
    
    for frac in unique_fractions:
        mask = renewable_fractions_rounded == frac
        frac_uncertainties = sigma_combined[mask]  # [n_frac_samples, n_buses]
        
        # Spatial predicted uncertainty: mean/std of predicted uncertainty over time for each bus
        # This measures: "How uncertain does the model think it is at each bus location?"
        spatial_predicted_uncertainty_mean = np.mean(frac_uncertainties, axis=0)  # [n_buses]
        spatial_predicted_uncertainty_std = np.std(frac_uncertainties, axis=0)   # [n_buses]
        
        # Temporal predicted uncertainty: mean/std of predicted uncertainty across buses for each timestep
        # This measures: "How uncertain does the model think it is at each time point?"
        temporal_predicted_uncertainty_mean = np.mean(frac_uncertainties, axis=1)  # [n_frac_samples]
        temporal_predicted_uncertainty_std = np.std(frac_uncertainties, axis=1)   # [n_frac_samples]
        
        uncertainty_data[round(float(frac), 1)] = {
            'spatial_predicted_mean': spatial_predicted_uncertainty_mean,
            'spatial_predicted_std': spatial_predicted_uncertainty_std,
            'temporal_predicted_mean': temporal_predicted_uncertainty_mean,
            'temporal_predicted_std': temporal_predicted_uncertainty_std,
            'mean_spatial_predicted': np.mean(spatial_predicted_uncertainty_mean),
            'max_spatial_predicted': np.max(spatial_predicted_uncertainty_mean),
            'mean_temporal_predicted': np.mean(temporal_predicted_uncertainty_mean),
            'max_temporal_predicted': np.max(temporal_predicted_uncertainty_mean)
        }
    
    return uncertainty_data


def calculate_uncertainty_metrics(predictions: np.ndarray, targets: np.ndarray, 
                                  renewable_fractions: np.ndarray, bus_types: np.ndarray = None) -> Dict:
    """
    Calculate uncertainty metrics for each renewable fraction.
    
    OPF Mode: Predictions are bus-type dependent unknowns:
    - PQ buses: [V, θ] (voltage magnitude, angle)
    - PV buses: [Q, θ] (reactive power, angle)
    - Slack buses: [P, Q] (active power, reactive power)
    
    Args:
        predictions: Shape [n_samples, n_buses, 2] - OPF unknowns
        targets: Shape [n_samples, n_buses, 2] - True OPF unknowns
        renewable_fractions: Shape [n_samples] - renewable fraction for each sample
        bus_types: Shape [n_samples, n_buses] - bus type codes [0=PQ, 1=PV, 2=Slack] (optional)
    
    Returns:
        Dictionary containing uncertainty metrics for each renewable fraction
    """
    # OPF Mode: Use both features (unknowns vary by bus type)
    # For simplicity, compute uncertainty as combined error across both features
    # This captures uncertainty in the predicted unknowns regardless of bus type
    
    # Calculate per-feature errors
    errors_feat0 = predictions[:, :, 0] - targets[:, :, 0]  # [n_samples, n_buses]
    errors_feat1 = predictions[:, :, 1] - targets[:, :, 1]  # [n_samples, n_buses]
    
    # Combined error magnitude (Euclidean norm per bus)
    # This captures total prediction error regardless of which unknowns are being predicted
    errors_combined = np.sqrt(errors_feat0**2 + errors_feat1**2)  # [n_samples, n_buses]
    
    # Get unique renewable fractions
    # This ensures keys like 0.2 don't become 0.19999999 or 0.20000001
    renewable_fractions_rounded = np.round(renewable_fractions, decimals=1)
    unique_fractions = np.unique(renewable_fractions_rounded)
    
    uncertainty_data = {}
    
    for frac in unique_fractions:
        # Get indices for this fraction (using rounded values for stable comparison)
        mask = renewable_fractions_rounded == frac
        frac_errors = errors_combined[mask]  # [n_frac_samples, n_buses]
        
        # Spatial uncertainty: std across time for each bus
        # This measures how much prediction error varies over time at each bus location
        spatial_uncertainty = np.std(frac_errors, axis=0)  # [n_buses]
        
        # Temporal uncertainty: mean error across buses for each timestep
        # This measures system-wide prediction error at each time point
        temporal_uncertainty = np.mean(frac_errors, axis=1)  # [n_frac_samples]
        
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
    fig, ax = plt.subplots(figsize=(12, 6))  # Match data profile story aspect ratio
    
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
    
    # Color map for different renewable fractions (match data profile story style)
    colors = plt.cm.viridis(np.linspace(0, 1, len(fractions)))
    
    # Always use time-series mode (hours of day on x-axis)
    hours_per_day = getattr(config, 'HOURS_PER_DAY', 24) if config else 24
    
    # Compute hourly statistics (mean and std) for each renewable fraction (match data profile style)
    for frac, color in zip(fractions, colors):
        temporal_unc = uncertainty_data[frac]['temporal']
        n_points = len(temporal_unc)
        
        # Map timesteps to hours of day (modulo 24) to show daily cycle pattern
        x_values = np.arange(n_points) % hours_per_day
        
        # Compute hourly mean and std (match data profile story style)
        hourly_mean = []
        hourly_std = []
        for h in range(hours_per_day):
            hour_mask = (x_values == h)
            if np.any(hour_mask):
                hourly_mean.append(np.mean(temporal_unc[hour_mask]))
                hourly_std.append(np.std(temporal_unc[hour_mask]))
            else:
                hourly_mean.append(np.nan)
                hourly_std.append(np.nan)
        
        hourly_mean = np.array(hourly_mean)
        hourly_std = np.array(hourly_std)
        
        # Plot with fill_between (match data profile story style)
        ax.plot(range(hours_per_day), hourly_mean, 
               label=f'{int(frac*100)}% Renewables',
               color=color, linewidth=2)
        ax.fill_between(range(hours_per_day), hourly_mean - hourly_std, 
                       hourly_mean + hourly_std, color=color, alpha=0.2)
    
    # Labels and title (match data profile story style)
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_xticks(range(0, hours_per_day, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, hours_per_day, 3)])
    ax.set_ylabel('Mean System Uncertainty σ_t (p.u.)', fontsize=11)
    
    title = f'Temporal Uncertainty - {case_name.upper()}'
    if model_name:
        title += f' - {model_name}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Legend and grid (match data profile story style)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    # Consolidated output - printed once at the end


def generate_uncertainty_visualizations(predictions: np.ndarray, targets: np.ndarray,
                                       renewable_fractions: np.ndarray, case_name: str,
                                       output_dir: str, model_name: str = "", config=None,
                                       bus_types: np.ndarray = None, model_outputs: np.ndarray = None):
    """
    Main function to generate all uncertainty visualizations.
    
    Args:
        predictions: Model predictions [n_samples, n_buses, 2] - OPF unknowns
        targets: True values [n_samples, n_buses, 2] - OPF unknowns
        renewable_fractions: Renewable fraction for each sample [n_samples]
        case_name: Test case name (e.g., "case33")
        output_dir: Directory to save outputs
        model_name: Optional model name for titles and filenames
        config: Optional config object for time-series mode detection
        bus_types: Optional [n_samples, n_buses] bus type array for OPF mode
        model_outputs: Optional [n_samples, n_buses, 4] - Full model outputs in heteroscedastic mode
                      [var1_pred, var2_pred, log_sigma_var1, log_sigma_var2]
    
    Returns:
        uncertainty_data: Dictionary with all calculated metrics
    """
    # Calculate error statistics (what actually happened)
    error_statistics = calculate_uncertainty_metrics(predictions, targets, renewable_fractions, bus_types=bus_types)
    
    # Calculate predicted uncertainties (what model thinks) if heteroscedastic mode
    predicted_uncertainties = None
    if model_outputs is not None:
        use_heteroscedastic = getattr(config, 'USE_HETEROSCEDASTIC_UNCERTAINTY', False) if config else False
        if use_heteroscedastic and model_outputs.shape[2] == 4:
            min_sigma = getattr(config, 'HETEROSCEDASTIC_MIN_SIGMA', 0.01) if config else 0.01
            max_sigma = getattr(config, 'HETEROSCEDASTIC_MAX_SIGMA', 10.0) if config else 10.0
            predicted_uncertainties = calculate_predicted_uncertainty_metrics(
                model_outputs, renewable_fractions, min_sigma=min_sigma, max_sigma=max_sigma
            )
    
    # Use predicted uncertainties if available, otherwise fall back to error statistics
    # Normalize keys for plotting functions (they expect 'spatial' and 'temporal')
    if predicted_uncertainties is not None:
        # Convert predicted uncertainty keys to match plotting function expectations
        uncertainty_data = {}
        for frac, data in predicted_uncertainties.items():
            uncertainty_data[frac] = {
                'spatial': data['spatial_predicted_mean'],  # Use mean predicted uncertainty
                'temporal': data['temporal_predicted_mean'],  # Use mean predicted uncertainty
                'mean_spatial': data['mean_spatial_predicted'],
                'max_spatial': data['max_spatial_predicted'],
                'mean_temporal': data['mean_temporal_predicted']
            }
    else:
        # Use error statistics (already has correct keys)
        uncertainty_data = error_statistics
    
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
