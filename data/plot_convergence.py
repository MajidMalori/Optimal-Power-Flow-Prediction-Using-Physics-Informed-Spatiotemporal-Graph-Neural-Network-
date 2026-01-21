"""
Professional Convergence Story Visualization
Shows data generation quality across renewable penetration levels
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_convergence_data(data_dir: str, case_name: str):
    """Load all convergence/audit JSON files for a case."""
    pattern = os.path.join(data_dir, f'{case_name}_*_frac*.json')
    files = glob.glob(pattern)
    
    audits = {}
    for f in files:
        filename = os.path.basename(f)
        parts = filename.replace('.json', '').split('_')
        
        ren_frac = None
        for part in parts:
            if part.startswith('frac'):
                try:
                    ren_frac = float(part[4:])
                    break
                except:
                    pass
        
        if ren_frac is not None:
            with open(f, 'r') as file:
                audits[ren_frac] = json.load(file)
    
    return audits


def plot_convergence_story(data_dir: str, case_name: str, output_path: str, config=None) -> str:
    """
    Create professional convergence story visualization.
    
    Shows success rate and resolution methods across renewable penetration levels.
    """
    audits = load_convergence_data(data_dir, case_name)
    
    if not audits:
        print(f"No convergence data found for {case_name}")
        return None
    
    # Extract data
    data_rows = []
    for ren_frac, audit in audits.items():
        # Handle both new and legacy formats
        stats = audit.get('raw_convergence_stats', audit)
        
        total = stats.get('total_timesteps', 0)
        successful = stats.get('successful', 0)
        failed = stats.get('failed', 0)
        success_rate = (successful / total * 100) if total > 0 else 0.0
        
        resolution = stats.get('resolution_methods', {})
        
        # Calculate implicit generator trips (successful but not recorded in other categories)
        known_sum = (
            resolution.get('strict_normal', 0) +
            resolution.get('strict_contingency', 0) +
            resolution.get('relaxed_contingency', 0) +
            resolution.get('restored_line', 0) +
            resolution.get('hard_reset', 0)
        )
        # Any successful timestep not accounted for is a generator trip
        trip_count = max(0, successful - known_sum)
        
        data_rows.append({
            'ren_pct': ren_frac * 100,
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': success_rate,
            'strict_normal': resolution.get('strict_normal', 0),
            'strict_contingency': resolution.get('strict_contingency', 0),
            'relaxed': resolution.get('relaxed_contingency', 0),
            'restored': resolution.get('restored_line', 0),
            'trip': trip_count,
            'hard_reset': resolution.get('hard_reset', 0),
        })
    
    df = pd.DataFrame(data_rows).sort_values('ren_pct')
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== LEFT: Success Rate Line Plot ==========
    ax1 = axes[0]
    
    ax1.plot(df['ren_pct'], df['success_rate'], 'o-', 
            linewidth=3, markersize=12, color='#2ecc71',
            markeredgewidth=2.5, markeredgecolor='white', 
            markerfacecolor='#2ecc71', label='Success Rate')
    
    # Add reference line at 100%
    ax1.axhline(100, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Perfect (100%)')
    
    # Add value labels
    for _, row in df.iterrows():
        ax1.text(row['ren_pct'], row['success_rate'] + 0.15, 
                f"{row['success_rate']:.2f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax1.set_xlabel('Renewable Penetration (%)', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=13)
    ax1.set_title('Convergence Success Rate', fontweight='bold', fontsize=14, pad=15)
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylim(max(97, df['success_rate'].min() - 0.5), 100.3)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower left', frameon=True, shadow=True)
    
    # ========== RIGHT: Resolution Methods Stacked Bar ==========
    ax2 = axes[1]
    
    # Prepare stacked data
    methods = ['strict_normal', 'strict_contingency', 'relaxed', 'restored', 'trip', 'hard_reset', 'failed']
    labels = ['Strict\n(Normal)', 'Strict\n(Contingency)', 'Relaxed', 'Line\nRestored', 'Gen\nTrip', 'Hard\nReset', 'Failed']
    # Professional high-contrast palette
    # Strict (Normal): Green (Success)
    # Strict (Contingency): Dark Teal (Success variant)
    # Relaxed: Yellow (Warning) - Distinct from Orange
    # Restored: Blue (Info) - Distinct from Orange/Red
    # Gen Trip: Purple (Special case)
    # Hard Reset: Orange (Severe Warning)
    # Failed: Red (Failure)
    colors_stack = ['#2ecc71', '#16a085', '#f1c40f', '#3498db', '#9b59b6', '#e67e22', '#c0392b']
    
    x = np.arange(len(df))
    width = 0.6
    
    bottom = np.zeros(len(df))
    for method, label, color in zip(methods, labels, colors_stack):
        values = df[method].values
        ax2.bar(x, values, width, bottom=bottom, label=label, 
               color=color, edgecolor='white', linewidth=1.5, alpha=0.85)
        
        # Add labels for non-zero segments
        for i, val in enumerate(values):
            if val > 0:
                y_pos = bottom[i] + val / 2
                ax2.text(i, y_pos, f'{int(val)}', ha='center', va='center',
                        fontweight='bold', fontsize=9, color='white')
        
        bottom += values
    
    ax2.set_xlabel('Renewable Penetration (%)', fontweight='bold', fontsize=13)
    ax2.set_ylabel('Number of Timesteps', fontweight='bold', fontsize=13)
    ax2.set_title('Resolution Methods Distribution', fontweight='bold', fontsize=14, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(p)}%' for p in df['ren_pct']])
    ax2.legend(loc='upper left', frameon=True, shadow=True, ncol=2, bbox_to_anchor=(1, 1))
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Main title
    case_num = case_name.replace('case', '')
    fig.suptitle(f'Convergence Quality Report - {case_num}-Bus System',
                 fontsize=17, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path
