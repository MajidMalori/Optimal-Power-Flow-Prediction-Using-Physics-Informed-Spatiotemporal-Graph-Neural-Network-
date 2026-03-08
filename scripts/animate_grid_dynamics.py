"""
Dynamic Grid Animation Script
Generates a GIF or MP4 showing the evolution of the grid topology and nodal voltages.
"""

import os
import glob
import json
import numpy as np
import networkx as nx
import pandapower as pp
import pandapower.topology as ppt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.processing.topology import load_network
from src.constants import FeatureIndices

# Professional styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.facecolor': '#f8f9fa',
    'figure.facecolor': '#f8f9fa'
})

def create_animation(case_name: str, data_dir: str, output_path: str, frac: float = 1.0, fps: int = 5, quiet: bool = False):
    """
    Creates an animated graph of the power system over time.
    """
    
    # 1. Load Base Network & Build Graph
    net = load_network(case_name)
    if net is None: return None
    
    G = ppt.create_nxgraph(net, include_lines=True, include_trafos=True, include_impedances=False)
    
    # Try to get geographical coordinates, fallback to spring layout
    pos = {}
    if hasattr(net, 'bus_geodata') and not net.bus_geodata.empty:
        for idx, row in net.bus_geodata.iterrows():
            pos[idx] = (row.x, row.y)
    else:
        # Generate consistent layout
        pos = nx.spring_layout(G, seed=42, k=0.5)

    # 2. Load Simulation Data (Voltages + Active Lines)
    feat_pattern = os.path.join(data_dir, f'{case_name}_features_frac*.npy')
    feat_files = sorted(glob.glob(feat_pattern))
    
    # Load specific fraction data
    feat_file = None
    for f in feat_files:
        if f"frac{frac}" in f:
            feat_file = f
            break
    if not feat_file and feat_files:
        feat_file = feat_files[-1]
        
    if not feat_file:
        if not quiet:
            print(f"No feature data found for {case_name}")
        return None
        
    features = np.load(feat_file)
    n_timesteps = features.shape[0]
    n_buses = features.shape[1]
    
    # Voltages
    voltages = features[:, :, FeatureIndices.VM]
    
    # Load Audit Data to reconstruct line exact states (which lines are open/closed)
    audit_file = feat_file.replace('_features_', '_data_quality_audit_').replace('.npy', '.json')
    if not os.path.exists(audit_file):
        if not quiet:
            print("Audit file missing, cannot accurately reconstruct topology swaps.")
        return None
        
    with open(audit_file, 'r') as f:
        audit = json.load(f)
        
    base_in_service = net.line.in_service.values.copy()
    n_lines = len(net.line)
    
    # Build complete state matrix [t, num_lines]
    line_states = np.zeros((n_timesteps, n_lines), dtype=int)
    for t in range(n_timesteps):
        line_states[t, :] = base_in_service
        
    for event in audit.get('switching_events', []):
        t = event['t']
        if t < n_timesteps:
            opened = event['opened']
            closed = event['closed']
            if opened < n_lines: line_states[t, opened] = False
            if closed < n_lines: line_states[t, closed] = True
            
            # Carry state forward until next switch (simplified assumption, real behavior might revert)
            # For strict visualization of the *event*, we'll highlight it.

    # 3. Setup Animation Figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Nodal Colors (Voltage) Setup
    vmin, vmax = 0.90, 1.10 # Physical bounds
    if case_name == 'case33': vmin, vmax = 0.85, 1.15
    cmap = plt.cm.get_cmap('RdYlBu') # Red (Low V), Yellow (Nominal), Blue (High V)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Bus Voltage (p.u.)', fontweight='bold')
    
    def update(frame):
        ax.clear()
        t = frame
        
        # Determine Edge Status for Timestep t
        current_state = line_states[t]
        
        active_edges = []
        inactive_edges = []
        
        # We need to map PandaPower line index to NetworkX edge
        # NetworkX edges from pandapower are tuples of bus indices
        for i, row in net.line.iterrows():
            u, v = row.from_bus, row.to_bus
            if current_state[i]:
                active_edges.append((u, v))
            else:
                inactive_edges.append((u, v))
                
        # Also include transformers (always active for this sim)
        for i, row in net.trafo.iterrows():
            u, v = row.hv_bus, row.lv_bus
            active_edges.append((u, v))

        # Title formatting
        hour = (t % 24)
        minute = int(((t % 1) * 60))
        time_str = f"{hour:02d}:{minute:02d}"
        ax.set_title(f'Dynamic Grid Topology & Voltage Profile\n{case_name.upper()} | Penetration: {frac*100}% | Timestep: {t} ({time_str})', 
                    fontweight='bold', fontsize=16, pad=20)
        
        # Draw Nodes (colored by voltage)
        v_current = voltages[t, :]
        node_colors = [cmap(norm(v)) for v in v_current]
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=150, edgecolors='black', linewidths=1.0)
        
        # Draw Edges (Active vs Inactive)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=active_edges, width=2.0, alpha=0.9, edge_color='#2c3e50')
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=inactive_edges, width=1.0, alpha=0.3, style='dashed', edge_color='#e74c3c')
        
        # Legend for edges
        custom_lines = [
            Line2D([0], [0], color='#2c3e50', lw=2),
            Line2D([0], [0], color='#e74c3c', lw=1, linestyle='--')
        ]
        ax.legend(custom_lines, ['Active Line / Trafo', 'Open / Disconnected Line'], 
                  loc='lower right', frameon=True, shadow=True, title="Topology Status")
        
        ax.axis('off')
        
    ani = animation.FuncAnimation(fig, update, frames=n_timesteps, interval=1000/fps)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as GIF
    gif_path = output_path.replace('.mp4', '.gif')
    try:
        ani.save(gif_path, writer='pillow', fps=fps)
    except Exception as e:
        if not quiet:
            print(f"Failed to save GIF: {e}")
        plt.close()
        return None
        
    plt.close()
    return gif_path

if __name__ == "__main__":
    import argparse
    import yaml
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='case33', help="Case name or 'all'")
    parser.add_argument('--frac', type=float, default=None, help="Single fraction (ignored when --case all)")
    parser.add_argument('--fps', type=int, default=5)
    args = parser.parse_args()

    # Load config for case list and fractions
    config_path = os.path.join(root_dir, 'configs', 'data_generation.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    fractions = config.get('renewable_fractions_to_run', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Determine cases to animate
    if args.case.lower() == 'all':
        cases = config.get('test_cases', ['case33', 'case57', 'case118'])
    else:
        cases = [args.case if args.case.startswith('case') else f'case{args.case}']
        if args.frac is not None:
            fractions = [args.frac]

    BUS_LABEL = {'case33': '33-bus', 'case57': '57-bus', 'case118': '118-bus'}

    for case in cases:
        n_buses = BUS_LABEL.get(case, case)
        data_dir = os.path.join(root_dir, 'data', '01_raw', case)
        out_dir  = os.path.join(root_dir, 'reports', 'animations', case)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{n_buses} | {len(fractions)} fractions")
        pbar = tqdm(
            fractions,
            desc=f"Anim {case} (frac   0%)",
            bar_format="{desc}: : {percentage:3.0f}%|{bar}| {n}/{total} fractions",
            leave=True
        )
        for frac in pbar:
            pbar.set_description(f"Anim {case} (frac {int(frac*100):3d}%)")
            out_path = os.path.join(out_dir, f'animation_{case}_frac{frac:.1f}.gif')
            create_animation(case, data_dir, out_path, frac=frac, fps=args.fps, quiet=True)
