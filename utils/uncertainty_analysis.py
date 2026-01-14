"""
Uncertainty Quantification and Visualization for Power System State Estimation.
Generates spatial and temporal uncertainty visualizations for trained models.

NOTE: Updated for Full State Reconstruction (10 outputs) with MC Dropout uncertainty.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster plotting
import matplotlib.pyplot as plt
# Optimize matplotlib for performance
plt.ioff()  # Turn off interactive mode (faster)
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import networkx as nx
import pandapower as pp
import pandapower.networks as pn
from typing import Dict, Tuple, List
import torch


# Cache for network topology (loaded once, reused for all plots)
_network_topology_cache = {}

def load_network_topology(case_name: str) -> Tuple[pp.pandapowerNet, nx.Graph, Dict]:
    """
    Load network topology and extract bus positions for visualization.
    PERFORMANCE OPTIMIZED: Caches topology to avoid reloading for each plot.
    
    Args:
        case_name: Name of the test case (e.g., "case33", "case57", "case118")
    
    Returns:
        net: Pandapower network object
        G: NetworkX graph
        pos: Dictionary of bus positions {bus_id: (x, y)}
    """
    # Check cache first (major performance improvement)
    if case_name in _network_topology_cache:
        return _network_topology_cache[case_name]
    
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
    
    # OPTIMIZED: Reduce iterations for faster layout computation
    # Spring layout is expensive - use fewer iterations (still looks good)
    if case_name == "case33":
        # For case33, use fewer iterations (small system, fast)
        pos = nx.spring_layout(G, seed=42, k=2, iterations=30)  # Reduced from 50
    elif case_name == "case57":
        # For case57, moderate iterations
        pos = nx.spring_layout(G, seed=42, k=1, iterations=40)  # Reduced from 100
    else:
        # For case118, use fewer iterations (large system, slow)
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)  # Reduced from 100
    
    # Cache the result
    _network_topology_cache[case_name] = (net, G, pos)
    
    return net, G, pos


def calculate_predicted_uncertainty_metrics(model_outputs: np.ndarray, renewable_fractions: np.ndarray,
                                            min_sigma: float = 0.01, max_sigma: float = 10.0,
                                            timesteps: np.ndarray = None) -> Dict:
    """
    Calculate PREDICTED uncertainty metrics from MC Dropout model outputs.
    
    This extracts the model's predicted uncertainties (std dev from MC Dropout) and analyzes them
    spatially and temporally. This is different from calculate_uncertainty_metrics which
    uses actual error statistics.
    
    Args:
        model_outputs: Shape [n_samples, n_buses, 10] - MC Dropout uncertainty (std dev) for full state
        renewable_fractions: Shape [n_samples] - renewable fraction for each sample
        min_sigma: Minimum sigma value for clamping (unused, kept for compatibility)
        max_sigma: Maximum sigma value for clamping (unused, kept for compatibility)
        timesteps: Optional timestep indices
    
    Returns:
        Dictionary containing predicted uncertainty metrics for each renewable fraction
    """
    # MC Dropout: model_outputs should be the std dev from multiple forward passes
    # For now, assume model_outputs is [n_samples, n_buses, 10] with uncertainty estimates
    # We'll focus on VM (col 8) and VA (col 9) uncertainties
    
    if model_outputs.shape[2] != 10:
        raise ValueError(f"Expected model_outputs shape [n_samples, n_buses, 10], got {model_outputs.shape}")
    
    # Extract VM and VA uncertainties
    sigma_vm = model_outputs[:, :, 8]  # [n_samples, n_buses]
    sigma_va = model_outputs[:, :, 9]  # [n_samples, n_buses]
    
    # Combined uncertainty: Use RMS for VM and VA
    sigma_combined = np.sqrt((sigma_vm**2 + sigma_va**2) / 2.0)  # [n_samples, n_buses] - RMS
    
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
        
        # Store timesteps for this fraction if available
        if timesteps is not None:
            frac_timesteps = timesteps[mask]
            uncertainty_data[round(float(frac), 1)]['timesteps'] = frac_timesteps
    
    return uncertainty_data


def calculate_uncertainty_metrics(predictions: np.ndarray, targets: np.ndarray, 
                                  renewable_fractions: np.ndarray, bus_types: np.ndarray = None,
                                  timesteps: np.ndarray = None) -> Dict:
    """
    Calculate uncertainty metrics for each renewable fraction (Full State Reconstruction).
    
    Focuses on VM (col 8) and VA (col 9) errors for all buses.
    
    Args:
        predictions: Shape [n_samples, n_buses, 10] - Full state predictions
        targets: Shape [n_samples, n_buses, 10] - True full state
        renewable_fractions: Shape [n_samples] - renewable fraction for each sample
        bus_types: Shape [n_samples, n_buses] - bus type codes (unused, kept for compatibility)
        timesteps: Shape [n_samples] - actual timestep indices for each sample (optional)
    
    Returns:
        Dictionary containing uncertainty metrics for each renewable fraction
    """
    if predictions.shape[2] != 10 or targets.shape[2] != 10:
        raise ValueError(f"Expected shape [n_samples, n_buses, 10], got predictions: {predictions.shape}, targets: {targets.shape}")
    
    # Calculate VM and VA errors
    errors_vm = predictions[:, :, 8] - targets[:, :, 8]  # [n_samples, n_buses]
    errors_va = predictions[:, :, 9] - targets[:, :, 9]  # [n_samples, n_buses]
    
    # Get unique renewable fractions
    renewable_fractions_rounded = np.round(renewable_fractions, decimals=1)
    unique_fractions = np.unique(renewable_fractions_rounded)
    
    uncertainty_data = {}
    
    for frac in unique_fractions:
        # Get indices for this fraction
        mask = renewable_fractions_rounded == frac
        frac_errors_vm = errors_vm[mask]  # [n_frac_samples, n_buses]
        frac_errors_va = errors_va[mask]  # [n_frac_samples, n_buses]
        
        # Calculate combined error (VM + VA)
        errors_combined = np.sqrt(frac_errors_vm**2 + frac_errors_va**2)  # [n_frac_samples, n_buses]
        spatial_uncertainty_combined = np.std(errors_combined, axis=0)  # [n_buses]
        temporal_uncertainty_combined = np.mean(errors_combined, axis=1)  # [n_frac_samples]
        
        frac_data = {
            'combined': {
                'spatial': spatial_uncertainty_combined,
                'temporal': temporal_uncertainty_combined,
                'mean_spatial': np.mean(spatial_uncertainty_combined),
                'max_spatial': np.max(spatial_uncertainty_combined),
                'mean_temporal': np.mean(temporal_uncertainty_combined)
            }
        }
        
        # Store timesteps for this fraction if available
        if timesteps is not None:
            frac_timesteps = timesteps[mask]
            frac_data['timesteps'] = frac_timesteps
        
        uncertainty_data[round(float(frac), 1)] = frac_data
    
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
    
    # Create figure with 2x3 grid (reduced size for faster rendering)
    fig = plt.figure(figsize=(14, 9))  # Reduced from 18x12 for faster rendering
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
            
            # OPTIMIZED: Draw network with reduced detail for faster rendering
            # Use simpler drawing options to speed up
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1.5, edge_color='gray', arrows=False)
            
            # Draw nodes colored by uncertainty (optimized node size)
            nodes = nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_color=spatial_unc,
                node_size=300,  # Reduced from 500 for faster rendering
                cmap='YlOrRd',
                vmin=vmin,
                vmax=vmax,
                edgecolors='black',
                linewidths=1.0  # Reduced from 1.5
            )
            
            # OPTIMIZED: Only draw labels for larger systems (skip for case118 to save time)
            if case_name != "case118":
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight='bold')
            
            # Title
            ax.set_title(f'{int(frac*100)}% Renewables\n(Mean σ: {uncertainty_data[frac]["mean_spatial"]:.4f} p.u.)',
                        fontsize=12, fontweight='bold')
        else:
            # Data missing - show placeholder (shouldn't happen with blocked_timeseries split)
            ax.text(0.5, 0.5, f'{int(frac*100)}% Renewables\n(No data available)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
            ax.set_title(f'{int(frac*100)}% Renewables', fontsize=12, fontweight='bold')
        
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
    
    # Save with optimized DPI (150 is high quality but 4x faster than 300)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')  # Close all figures to free memory
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
    # Reduced figure size for faster rendering
    fig, ax = plt.subplots(figsize=(10, 5))  # Reduced from 12x6 for faster rendering
    
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
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close('all')  # Close all figures to free memory
        return
    
    # Continue with available fractions
    expected_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    missing_fractions = [f for f in expected_fractions if f not in fractions]
    
    # Warn about missing fractions (shouldn't happen with blocked_timeseries split, but check anyway)
    if missing_fractions:
        missing_pct = [int(f*100) for f in missing_fractions]
        print(f"[Uncertainty] Missing fractions in test set: {missing_pct}%")
    
    # Color map for different renewable fractions (match data profile story style)
    colors = plt.cm.viridis(np.linspace(0, 1, len(fractions)))
    
    # Always use time-series mode (hours of day on x-axis)
    hours_per_day = getattr(config, 'HOURS_PER_DAY', 24) if config else 24
    
    # Compute hourly statistics (mean and std) for each renewable fraction (match data profile style)
    for frac, color in zip(fractions, colors):
        temporal_unc = uncertainty_data[frac]['temporal']
        n_points = len(temporal_unc)
        
        # Use actual timesteps if available, otherwise fall back to array indices
        if 'timesteps' in uncertainty_data[frac] and uncertainty_data[frac]['timesteps'] is not None:
            # Use actual timesteps to compute hours of day
            actual_timesteps = uncertainty_data[frac]['timesteps']
            x_values = actual_timesteps % hours_per_day
        else:
            # Fallback: Map array indices to hours (assumes first sample is hour 0)
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
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save with optimized DPI (150 is high quality but 4x faster than 300)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close('all')  # Close all figures to free memory
    # Consolidated output - printed once at the end


def generate_uncertainty_visualizations(predictions: np.ndarray, targets: np.ndarray,
                                       renewable_fractions: np.ndarray, case_name: str = "",
                                       output_dir: str = "", model_name: str = "", config=None,
                                       bus_types: np.ndarray = None, model_outputs: np.ndarray = None,
                                       timesteps: np.ndarray = None):
    """
    Main function to generate all uncertainty visualizations.
    
    Args:
        predictions: Model predictions [n_samples, n_buses, 10] - Full state
        targets: True values [n_samples, n_buses, 10] - Full state
        renewable_fractions: Renewable fraction for each sample [n_samples] - REQUIRED
        case_name: Test case name (e.g., "case33")
        output_dir: Directory to save outputs
        model_name: Optional model name for titles and filenames
        config: Optional config object for time-series mode detection
        bus_types: Optional [n_samples, n_buses] bus type array (unused, kept for compatibility)
        model_outputs: Optional [n_samples, n_buses, 10] - MC Dropout uncertainty (std dev)
        timesteps: Optional [n_samples] - actual timestep indices for temporal plotting
    
    Returns:
        uncertainty_data: Dictionary with all calculated metrics
    """
    # Fail fast if renewable_fractions is missing (required for grouping analysis)
    if renewable_fractions is None:
        raise ValueError("renewable_fractions is REQUIRED for uncertainty analysis. Extract it from test batch data.")
    if len(renewable_fractions) != predictions.shape[0]:
        raise ValueError(f"renewable_fractions length ({len(renewable_fractions)}) must match predictions batch size ({predictions.shape[0]})")
    
    # Calculate error statistics (what actually happened)
    error_statistics = calculate_uncertainty_metrics(predictions, targets, renewable_fractions, bus_types=bus_types, timesteps=timesteps)
    
    # Calculate predicted uncertainties (what model thinks) - MC Dropout mode
    predicted_uncertainties = None
    if model_outputs is not None:
        if model_outputs.shape[2] == 10:
            predicted_uncertainties = calculate_predicted_uncertainty_metrics(
                model_outputs, renewable_fractions, timesteps=timesteps
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
        # Use error statistics - extract 'combined' data for plotting (backward compatibility)
        # The new structure has bus-type-specific data, but plotting functions use 'combined'
        uncertainty_data = {}
        for frac, frac_data in error_statistics.items():
            if isinstance(frac_data, dict) and 'combined' in frac_data:
                # New structure with bus-type separation - use 'combined' for plotting
                uncertainty_data[frac] = frac_data['combined'].copy()
                # Preserve timesteps if available
                if 'timesteps' in frac_data:
                    uncertainty_data[frac]['timesteps'] = frac_data['timesteps']
            else:
                # Old structure (backward compatibility)
                uncertainty_data[frac] = frac_data
    
    # Ensure timesteps are passed through if available
    if timesteps is not None and isinstance(uncertainty_data, dict):
        # Timesteps should already be in error_statistics from calculate_uncertainty_metrics
        # But ensure they're preserved in the final uncertainty_data structure
        for frac in uncertainty_data.keys():
            if 'timesteps' not in uncertainty_data[frac] and timesteps is not None:
                # Extract timesteps for this fraction
                frac_mask = np.round(renewable_fractions, decimals=1) == frac
                if np.any(frac_mask):
                    uncertainty_data[frac]['timesteps'] = timesteps[frac_mask]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate spatial comparison grid
    spatial_output = os.path.join(output_dir, 'uncertainty_spatial.png')
    plot_spatial_comparison_grid(uncertainty_data, case_name, spatial_output, model_name)
    
    # Generate temporal comparison curves (hours if time-series mode)
    temporal_output = os.path.join(output_dir, 'uncertainty_temporal.png')
    plot_temporal_comparison_curves(uncertainty_data, case_name, temporal_output, model_name, config)
    
    # Plots saved silently
    
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
