"""
Professional Switching Heatmap Visualization
Generates a Time-Series Matrix of line statuses (Closed/Open).
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.processing.topology import load_network

# Professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})

def plot_switching_heatmap(data_dir: str, case_name: str, output_path: str, frac: float = 1.0) -> str:
    """
    Creates a heatmap of Line ID vs Timestep showing open/closed status.
    Uses the audit JSON to reconstruct line states.
    """
    # Load base network to get initial line status
    try:
        net = load_network(case_name)
        n_lines = len(net.line)
        base_status = net.line.in_service.values.astype(int) # 1 if closed, 0 if open
    except Exception as e:
        print(f"Failed to load network for {case_name}: {e}")
        return None

    # Load audit file
    audit_pattern = os.path.join(data_dir, f'{case_name}_data_quality_audit_frac*.json')
    audit_files = sorted(glob.glob(audit_pattern))
    if not audit_files:
        return None
    
    # Pick the audit file matching frac, or fallback to the last one
    audit_path = audit_files[-1]
    for af in audit_files:
        if f"frac{frac}" in af:
            audit_path = af
            break
            
    with open(audit_path, 'r') as f:
        audit = json.load(f)
        
    t_total = audit.get('total_timesteps', 96)
    events = audit.get('switching_events', [])
    
    # Initialize state matrix: [n_lines, t_total]
    # State: 0 = Open, 1 = Baseline Closed, 2 = Strategically Closed
    state_matrix = np.zeros((n_lines, t_total), dtype=int)
    
    for t in range(t_total):
        state_matrix[:, t] = base_status
        
    for event in events:
        t = event['t']
        if t < t_total:
            closed_idx = event['closed']
            opened_idx = event['opened']
            if opened_idx < n_lines:
                state_matrix[opened_idx, t] = 0
            if closed_idx < n_lines:
                state_matrix[closed_idx, t] = 2

    # Plot
    fig, ax = plt.subplots(figsize=(16, max(6, n_lines * 0.15)))
    
    # Custom colormap: 0: Light Gray, 1: Steel Blue, 2: Bright Orange
    cmap = mcolors.ListedColormap(['#ecf0f1', '#34495e', '#e67e22'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Heatmap
    cax = ax.imshow(state_matrix, aspect='auto', cmap=cmap, norm=norm, origin='lower', interpolation='none')
    
    ax.set_xlabel('Timestep', fontweight='bold')
    ax.set_ylabel('Transmission Line ID', fontweight='bold')
    ax.set_title(f'Spatiotemporal Grid Configuration Matrix - {case_name.upper()}\n(Illustrating Dynamic Rerouting & Tie-Line Utilization)', pad=15, fontweight='bold')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ecf0f1', edgecolor='#bdc3c7', label='Line Open (Disconnected)'),
        Patch(facecolor='#34495e', edgecolor='#2c3e50', label='Line Closed (Base Config)'),
        Patch(facecolor='#e67e22', edgecolor='#d35400', label='Strategic Tie-Line Closed')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.15), ncol=3, frameon=False)
    
    if n_lines < 60:
        ax.set_yticks(np.arange(-0.5, n_lines, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.set_yticks(np.arange(0, n_lines, 5))
        ax.set_yticklabels(np.arange(0, n_lines, 5))
    else:
        ax.set_yticks(np.arange(0, n_lines, 10))
        
    if t_total <= 100:
        ax.set_xticks(np.arange(0, t_total, 10))
    else:
        ax.set_xticks(np.arange(0, t_total, t_total//10))

    fig.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

if __name__ == "__main__":
    import sys
    case = sys.argv[1] if len(sys.argv) > 1 else 'case33'
    plot_switching_heatmap('data/raw', case, f'reports/raw_data/heatmap_{case}.png')
