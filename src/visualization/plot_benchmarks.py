import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_premium_aesthetics():
    """Set the aesthetic parameters for the plots and silence font warnings."""
    import logging
    import warnings
    # Silence matplotlib font manager warnings and general findfont warnings
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*findfont: Generic family.*")
    warnings.filterwarnings("ignore", message=".*findfont: Font family.*")

    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans' # Use a specific installed font
    plt.rcParams['figure.dpi'] = 300

SOLVER_DISPLAY = {
    "nr": "Newton-Raphson",
    "iwamoto_nr": "NR+Iwamoto",
    "gs": "Gauss-Seidel",
    "bfsw": "Backward/Fwd"
}

SOLVER_COLORS = {
    "nr": "#e63946",
    "iwamoto_nr": "#e76f51",
    "gs": "#2a9d8f",
    "bfsw": "#264653"
}

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

    # --- Plot 3: Multi-Solver Efficiency (Speedup vs All Solvers) ---
    solver_speeds = results[0].get('solver_speeds', {})
    if solver_speeds:
        fig, ax = plt.subplots(figsize=(12, 7))
        inf_times = [r['avg_inf_ms'] for r in results]
        
        # GNN inference bars
        bars = ax.bar(models, inf_times, color='#4cc9f0', label='GNN Inference', zorder=3)
        
        # Horizontal lines for each solver
        for alg, ms in solver_speeds.items():
            color = SOLVER_COLORS.get(alg, '#888888')
            name = SOLVER_DISPLAY.get(alg, alg)
            ax.axhline(y=ms, color=color, linestyle='--', linewidth=2, 
                       label=f'{name} ({ms:.1f} ms)', zorder=2)
        
        ax.set_yscale('log')
        ax.set_ylabel('Time per Sample (ms) - Log Scale', fontsize=12, fontweight='bold')
        ax.set_title(f"Multi-Solver Efficiency: {case_name}", fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Add speedup annotations (vs NR)
        nr_ms = solver_speeds.get("nr", 1.0)
        for i, r in enumerate(results):
            sp = nr_ms / r['avg_inf_ms']
            ax.text(i, r['avg_inf_ms'], f"{sp:.0f}x", ha='center', va='bottom', 
                    fontweight='bold', fontsize=9, zorder=4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "efficiency.png"), bbox_inches='tight')
        plt.close()

    # --- Plot 4: Solver Accuracy Comparison (GNN vs Classical Solvers vs NR Truth) ---
    solver_accuracy = results[0].get('solver_accuracy', {})
    if solver_accuracy:
        fig, (ax_vm, ax_va) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Collect all entries: solvers + GNN models
        all_labels = []
        all_mae_vm = []
        all_mae_va = []
        all_colors = []
        
        # Classical solvers first
        for alg, acc in solver_accuracy.items():
            all_labels.append(SOLVER_DISPLAY.get(alg, alg))
            all_mae_vm.append(acc['mae_vm'])
            all_mae_va.append(acc['mae_va'])
            all_colors.append(SOLVER_COLORS.get(alg, '#888888'))
        
        # GNN models
        gnn_palette = plt.cm.cool(np.linspace(0.2, 0.8, len(results)))
        for i, r in enumerate(results):
            all_labels.append(r['model'])
            all_mae_vm.append(r['mae_vm'])
            all_mae_va.append(r['mae_va'])
            all_colors.append(gnn_palette[i])
        
        x = np.arange(len(all_labels))
        
        # VM subplot
        ax_vm.barh(x, all_mae_vm, color=all_colors, edgecolor='white', linewidth=0.5)
        ax_vm.set_yticks(x)
        ax_vm.set_yticklabels(all_labels, fontsize=10)
        ax_vm.set_xlabel('MAE Voltage Magnitude (p.u.)', fontsize=11, fontweight='bold')
        ax_vm.set_title('VM Accuracy vs NR Ground Truth', fontsize=13, fontweight='bold')
        ax_vm.invert_yaxis()
        ax_vm.set_xscale('log')
        
        # VA subplot
        ax_va.barh(x, all_mae_va, color=all_colors, edgecolor='white', linewidth=0.5)
        ax_va.set_yticks(x)
        ax_va.set_yticklabels(all_labels, fontsize=10)
        ax_va.set_xlabel('MAE Voltage Angle (rad)', fontsize=11, fontweight='bold')
        ax_va.set_title('VA Accuracy vs NR Ground Truth', fontsize=13, fontweight='bold')
        ax_va.invert_yaxis()
        ax_va.set_xscale('log')
        
        fig.suptitle(f"Solver vs GNN Accuracy Comparison: {case_name}", 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "solver_comparison.png"), bbox_inches='tight')
        plt.close()
