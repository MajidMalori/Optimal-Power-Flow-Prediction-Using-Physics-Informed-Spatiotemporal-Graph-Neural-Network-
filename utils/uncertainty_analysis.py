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
    Uses proper hierarchical/radial layouts that match actual power system topology.
    
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
    
    # Add edges (lines) - THESE ARE THE ACTUAL POWER SYSTEM CONNECTIONS
    for _, line in net.line.iterrows():
        G.add_edge(line.from_bus, line.to_bus)
    
    # Generate LAYERED HIERARCHICAL layout for power networks
    # Handles disconnected components and meshed networks properly
    
    from collections import deque, defaultdict
    import math
    
    # Find all connected components
    components = list(nx.connected_components(G))
    
    # Sort components: largest first, or component containing bus 0 first
    def component_priority(comp):
        if 0 in comp:
            return (0, -len(comp))  # Bus 0 component first, then by size
        return (1, -len(comp))
    
    components = sorted(components, key=component_priority)
    
    pos = {}
    component_offset = 0
    
    for comp_idx, component in enumerate(components):
        # Create subgraph for this component
        subgraph = G.subgraph(component)
        
        # Find root node (bus 0 if in this component, otherwise lowest numbered node)
        if 0 in component:
            root = 0
        else:
            root = min(component)
        
        # Assign levels using BFS from root
        levels = {}
        parent = {}
        children = defaultdict(list)
        
        queue = deque([root])
        visited = {root}
        levels[root] = 0
        
        while queue:
            node = queue.popleft()
            for neighbor in sorted(subgraph.neighbors(node)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    if node not in children:
                        children[node] = []
                    children[node].append(neighbor)
                    levels[neighbor] = levels[node] + 1
                    queue.append(neighbor)
        
        # Group nodes by level
        level_nodes = defaultdict(list)
        for node, level in levels.items():
            level_nodes[level].append(node)
        
        # Position nodes using layered approach
        # Each level is a horizontal layer, nodes distributed evenly
        max_level = max(levels.values()) if levels else 0
        
        # Calculate width needed for this component
        max_nodes_at_level = max(len(nodes) for nodes in level_nodes.values()) if level_nodes else 1
        component_width = max_nodes_at_level * 4  # Horizontal spacing = 4 units
        
        # Position nodes level by level
        for level in range(max_level + 1):
            nodes_at_level = sorted(level_nodes[level])
            n_nodes = len(nodes_at_level)
            
            if n_nodes == 0:
                continue
            
            # Calculate x positions for this level
            if n_nodes == 1:
                # Single node - center it
                x_positions = [component_offset + component_width / 2]
            else:
                # Multiple nodes - distribute evenly
                spacing = component_width / (n_nodes + 1)
                x_positions = [component_offset + spacing * (i + 1) for i in range(n_nodes)]
            
            # Assign positions
            for node, x in zip(nodes_at_level, x_positions):
                y = -level * 5  # Vertical spacing = 5 units
                pos[node] = (x, y)
        
        # Update offset for next component
        component_offset += component_width + 15  # 15 units gap between components
    
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
        
        # Adjust node size based on number of buses
        num_buses = len(G.nodes())
        if num_buses <= 33:
            node_size = 350
            font_size = 9
        elif num_buses <= 57:
            node_size = 280
            font_size = 8
        else:
            node_size = 180
            font_size = 7
        
        # Draw edges FIRST (background layer)
        # The improved layouts ensure nodes don't overlap with edges
        nx.draw_networkx_edges(
            G, pos, ax=ax, 
            alpha=0.5, 
            width=2.5, 
            edge_color='#666666'
        )
        
        # Draw nodes colored by uncertainty
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=spatial_unc,
            node_size=node_size,
            cmap='YlOrRd',
            vmin=vmin,
            vmax=vmax,
            edgecolors='black',
            linewidths=2.0,
            alpha=0.95
        )
        
        # Add node labels
        labels = nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size, 
                                        font_weight='bold', font_color='black')
        
        # Draw edges AGAIN on top (foreground layer)
        # This ensures edges are visible even if they pass near nodes
        nx.draw_networkx_edges(
            G, pos, ax=ax, 
            alpha=0.25, 
            width=2.5, 
            edge_color='#333333'
        )
        
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
                                   output_path: str, model_name: str = "", config=None):
    """
    Generate temporal uncertainty comparison with 6 curves overlaid.
    
    Args:
        uncertainty_data: Dictionary with uncertainty metrics per renewable fraction
        case_name: Name of the test case
        output_path: Where to save the output image
        model_name: Optional model name for title
        config: Optional config object to check if using time-series mode
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort renewable fractions
    fractions = sorted(uncertainty_data.keys())
    
    # Color map for different renewable fractions
    colors = plt.cm.viridis(np.linspace(0, 1, len(fractions)))
    
    # Determine if we should use hours or timesteps for x-axis
    use_time_series = getattr(config, 'USE_TIME_SERIES', False) if config else False
    hours_per_day = getattr(config, 'HOURS_PER_DAY', 24) if config else 24
    
    for frac, color in zip(fractions, colors):
        temporal_unc = uncertainty_data[frac]['temporal']
        n_points = len(temporal_unc)
        
        if use_time_series:
            # TIME-SERIES MODE: X-axis shows hours (0-24)
            # Map timesteps to hours within a day
            hours = np.arange(n_points) % hours_per_day
            x_values = hours
        else:
            # MONTE CARLO MODE: X-axis shows timesteps
            x_values = np.arange(n_points)
        
        # Plot with label
        ax.plot(x_values, temporal_unc, 
               label=f'{int(frac*100)}% Renewables (μ={uncertainty_data[frac]["mean_temporal"]:.4f})',
               color=color, linewidth=2, alpha=0.8, marker='o', markersize=4)
    
    # Labels and title
    if use_time_series:
        ax.set_xlabel('Hour of Day', fontsize=14, fontweight='bold')
        # Set x-axis to show 0-24 hours
        ax.set_xlim(0, hours_per_day)
        ax.set_xticks(np.arange(0, hours_per_day+1, 3))  # Show every 3 hours
    else:
        ax.set_xlabel('Timestep', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Mean System Uncertainty σ_t (p.u.)', fontsize=14, fontweight='bold')
    
    title = f'Temporal Uncertainty Curve - {case_name.upper()}'
    if model_name:
        title += f' - {model_name}'
    if use_time_series:
        title += ' (Daily Cycle)'
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
    print(f"[Uncertainty] Saved temporal comparison: {output_path}")


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
    print(f"\n[Uncertainty] Generating uncertainty visualizations for {case_name} - {model_name}...")
    
    # Calculate uncertainty metrics
    uncertainty_data = calculate_uncertainty_metrics(predictions, targets, renewable_fractions)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate spatial comparison grid (topology-aware)
    spatial_output = os.path.join(output_dir, 'uncertainty_spatial_comparison.png')
    plot_spatial_comparison_grid(uncertainty_data, case_name, spatial_output, model_name)
    
    # Generate temporal comparison curves (hours if time-series mode)
    temporal_output = os.path.join(output_dir, 'uncertainty_temporal_comparison.png')
    plot_temporal_comparison_curves(uncertainty_data, case_name, temporal_output, model_name, config)
    
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

