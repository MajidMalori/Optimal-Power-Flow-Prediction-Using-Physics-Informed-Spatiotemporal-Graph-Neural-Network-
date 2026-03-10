import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import networkx as nx
import pandapower as pp
import pandapower.networks as pn
from typing import Dict, Tuple

# Cache for network topology
_network_topology_cache = {}

def set_premium_aesthetics():
    """Set the aesthetic parameters for the plots and silence font warnings."""
    import logging
    import warnings
    # Silence matplotlib font manager warnings and general findfont warnings
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*findfont: Generic family.*")
    warnings.filterwarnings("ignore", message=".*findfont: Font family.*")
    
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', # Use a specific installed font to avoid generic searches
        'axes.unicode_minus': False,
        'figure.dpi': 150
    })

def load_network_topology(case_name: str) -> Tuple[pp.pandapowerNet, nx.Graph, Dict]:
    """Load network topology and extract bus positions."""
    if case_name in _network_topology_cache:
        return _network_topology_cache[case_name]
    
    if case_name == "case33":
        net = pn.case33bw()
    elif case_name == "case57":
        net = pn.case57()
    elif case_name == "case118":
        net = pn.case118()
    else:
        # Fallback to loading via project utility if available, or error
        try:
             from src.processing.topology import load_network
             net = load_network(case_name)
        except:
             raise ValueError(f"Unknown case: {case_name}")
    
    G = nx.Graph()
    for bus_idx in net.bus.index:
        G.add_node(bus_idx)
    for _, line in net.line.iterrows():
        G.add_edge(line.from_bus, line.to_bus)
    
    # Fast layout computation
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
    
    _network_topology_cache[case_name] = (net, G, pos)
    return net, G, pos

def plot_spatial_comparison_grid(uncertainty_data: Dict, case_name: str, 
                                 output_path: str, model_name: str = ""):
    """Generate 6-panel spatial uncertainty comparison grid."""
    set_premium_aesthetics()
    net, G, pos = load_network_topology(case_name)
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    expected_fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Global color limits
    all_spatial = [data['spatial'] for data in uncertainty_data.values() if 'spatial' in data]
    if all_spatial:
        vmin = min(s.min() for s in all_spatial)
        vmax = max(s.max() for s in all_spatial)
        if vmin == vmax: vmax += 1e-4
    else:
        vmin, vmax = 0, 0.01

    for idx, frac in enumerate(expected_fractions):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        # Draw base topology
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1.5, edge_color='gray', arrows=False)
        
        if frac in uncertainty_data:
            spatial_unc = uncertainty_data[frac]['spatial']
            nodes = nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_color=spatial_unc,
                node_size=300,
                cmap='YlOrRd',
                vmin=vmin, vmax=vmax,
                edgecolors='black', linewidths=1.0
            )
            if case_name != "case118":
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_weight='bold')
            
            ax.set_title(f'{int(frac*100)}% Renewables\nμ: {uncertainty_data[frac]["mean_spatial"]:.4f} p.u.', 
                         fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'{int(frac*100)}% N/A', ha='center', va='center', transform=ax.transAxes)
            
        ax.axis('off')

    # Colorbar
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Predictive Uncertainty σ [TTA] (p.u.)', fontsize=12, fontweight='bold')
    
    fig.suptitle(f'Predictive Uncertainty (Model Doubt) - {case_name.upper()} - {model_name}', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_temporal_comparison_curves(uncertainty_data: Dict, case_name: str,
                                   output_path: str, model_name: str = ""):
    """Generate temporal uncertainty curves with shaded confidence bands."""
    set_premium_aesthetics()
    fig, ax = plt.subplots(figsize=(16, 6))
    
    fractions = sorted(uncertainty_data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(fractions)))
    
    for frac, color in zip(fractions, colors):
        if 'temporal_mean' not in uncertainty_data[frac]: continue
        
        mean_vals = np.array(uncertainty_data[frac]['temporal_mean'])
        std_vals = np.array(uncertainty_data[frac]['temporal_std'])
        
        # Map to 24 hours
        x = np.linspace(0, 24, len(mean_vals))
        
        # Thick, smooth line
        ax.plot(x, mean_vals, label=f'{int(frac*100)}% Renewables', 
                color=color, lw=2.5, alpha=0.9)
        
        # Shaded confidence band (visible even if std is small)
        ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, 
                         color=color, alpha=0.2)

    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean System Uncertainty σ_t (p.u.)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    ax.set_title(f'Temporal Uncertainty - {case_name.upper()} - {model_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

