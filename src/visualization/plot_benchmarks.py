import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_premium_aesthetics():
    """Set the aesthetic parameters for the plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']
    plt.rcParams['figure.dpi'] = 300

def plot_benchmark_results(results, case_name, output_dir):
    """
    Generate benchmark visualization plots.
    
    Args:
        results: List of result dictionaries from evaluation
        case_name: Name of the power system case
        output_dir: Directory to save the plots
    """
    if not results:
        print(f"No results to plot for {case_name}")
        return

    os.makedirs(output_dir, exist_ok=True)
    set_premium_aesthetics()
    
    models = [r['model'] for r in results]
    
    # --- Plot 1: Accuracy Benchmark (MAE) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, [r['mae_vm'] for r in results], width, label='Voltage Mag (p.u.)', color='#4361ee')
    ax1.set_ylabel('MAE (VM)', color='black', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, [r['mae_va'] for r in results], width, label='Voltage Angle (rad)', color='#3a0ca3')
    ax2.set_ylabel('MAE (VA)', color='black', fontsize=12, fontweight='bold')
    ax2.grid(False)

    plt.title(f"Accuracy Benchmark: {case_name}", fontsize=14, fontweight='bold', pad=20)
    fig.legend(loc='upper left', bbox_to_anchor=(1.02, 0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy.png"), bbox_inches='tight')
    plt.close()

    # --- Plot 2: Physics Gap (Constraint Satisfaction) ---
    plt.figure(figsize=(12, 7))
    metrics = ['p_sat', 'q_sat', 'v_sat', 's_sat']
    labels = ['Power P', 'Power Q', 'Voltage', 'Branch']
    colors = ['#f72585', '#7209b7', '#3a0ca3', '#4361ee']
    
    n_models = len(models)
    n_metrics = len(metrics)
    x = np.arange(n_models)
    width = 0.18
    
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [r[metric] * 100 for r in results]
        plt.bar(x + (i - n_metrics/2 + 0.5) * width, vals, width, label=label, color=color)

    plt.ylabel('Satisfaction Rate (%)', fontsize=12, fontweight='bold')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.title(f"The 'Physics Gap': Constraint Satisfaction on {case_name}", fontsize=14, fontweight='bold', pad=20)
    plt.legend(title="Constraints", loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "physics_gap.png"), bbox_inches='tight')
    plt.close()

    # --- Plot 3: Efficiency Gain (Speedup) ---
    plt.figure(figsize=(10, 6))
    inf_times = [r['avg_inf_ms'] for r in results]
    pp_times = [r['avg_pp_ms'] for r in results]
    
    plt.bar(models, inf_times, color='#4cc9f0', label='GNN Inference (ms)')
    plt.axhline(y=np.mean(pp_times), color='#f72585', linestyle='--', label='Pandapower IPOPT (avg)')
    
    plt.yscale('log')
    plt.ylabel('Time per Sample (ms) - Log Scale', fontsize=12, fontweight='bold')
    plt.title(f"Efficiency Gain: {case_name} (Speedup vs Reference)", fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=45, ha='right')
    
    for i, r in enumerate(results):
        plt.text(i, r['avg_inf_ms'], f"{r['speedup']:.1f}x", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency.png"), bbox_inches='tight')
    plt.close()
    
    print(f"Benchmark plots saved to {os.path.relpath(output_dir)}")
