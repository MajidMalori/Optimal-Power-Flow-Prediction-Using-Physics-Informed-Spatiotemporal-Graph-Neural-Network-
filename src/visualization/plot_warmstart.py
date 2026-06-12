import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Premium Aesthetic Configuration
THEME_COLORS = {
    "background": "#F8F9FA",
    "grid": "#E9ECEF",
    "text": "#212529",
    "palette": ["#ADB5BD", "#6C757D", "#FF4757"] # Neutral Gray, Dark Gray, Vibrant Primary (Ours)
}

def setup_premium_style():
    sns.set_theme(style="white", rc={
        "axes.facecolor": THEME_COLORS["background"],
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": THEME_COLORS["grid"],
        "axes.edgecolor": THEME_COLORS["grid"],
        "text.color": THEME_COLORS["text"],
        "axes.labelcolor": THEME_COLORS["text"],
        "xtick.color": THEME_COLORS["text"],
        "ytick.color": THEME_COLORS["text"],
        "font.sans-serif": ["Inter", "Roboto", "Helvetica", "Arial", "sans-serif"]
    })

def plot_warmstart_metrics(df, case_name, output_dir):
    """
    Generates premium, high-impact visualizations for warm-start benchmarking.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_premium_style()
    
    # Pre-map the hues for a clean, professional legend
    hue_order = ["Generic Flat Start (1.0 p.u., 0°)", "Linearized DC Start", "Physics-Informed GNN"]
    palette = {
        "Generic Flat Start (1.0 p.u., 0°)": THEME_COLORS["palette"][0],
        "Linearized DC Start": THEME_COLORS["palette"][1],
        "Physics-Informed GNN": THEME_COLORS["palette"][2]
    }

    # 1. Convergence Rate (Robustness Plot)
    plt.figure(figsize=(10, 5))
    success_df = df.groupby(["Model", "InitMethod"])["Success"].mean().reset_index()
    success_df["Success"] *= 100  # Convert to %
    
    ax = sns.barplot(data=success_df, x="Model", y="Success", hue="InitMethod", 
                     hue_order=hue_order, palette=palette, edgecolor="white", linewidth=1.5)
    
    plt.title(f"[{case_name.upper()}] Solver Robustness: Convergence Success Rate", weight='bold', pad=20)
    plt.ylabel("Convergence Rate (%)")
    plt.xlabel("")
    plt.ylim(0, 110)
    plt.xticks(rotation=15)
    plt.grid(axis='x')
    
    # Legend at bottom
    plt.legend(title="", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{case_name}_robustness.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Computation Time (Log Scale)
    plt.figure(figsize=(10, 6))
    df_valid = df[df['Success'] == True]
    ax = sns.boxplot(data=df_valid, x="Model", y="Time_ms", hue="InitMethod", 
                     hue_order=hue_order, palette=palette, fliersize=0, linewidth=1.2)
    
    # Add jittered points for detail - removed redundant 'label' to fix Seaborn bug
    sns.stripplot(data=df_valid, x="Model", y="Time_ms", hue="InitMethod", 
                  hue_order=hue_order, palette=palette, dodge=True, alpha=0.3, jitter=0.2, ax=ax)
    
    plt.yscale('log')
    plt.title(f"[{case_name.upper()}] Computational Efficiency: Solve Time (Log Scale)", weight='bold', pad=20)
    plt.ylabel("Computation Time (ms)")
    plt.xlabel("")
    plt.xticks(rotation=15)
    
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[:3], labels[:3], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{case_name}_solve_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Iteration Count
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df_valid, x="Model", y="Iterations", hue="InitMethod", 
                     hue_order=hue_order, palette=palette, errorbar=None, edgecolor="white")
    
    # Annotate with mean values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=THEME_COLORS["text"])

    plt.title(f"[{case_name.upper()}] Algorithm Efficiency: Iterations to Convergence", weight='bold', pad=20)
    plt.ylabel("Avg. Newton-Raphson Iterations")
    plt.xlabel("")
    plt.xticks(rotation=15)
    
    plt.legend(title="", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{case_name}_iterations.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cross_case_scaling(df_global, output_dir):
    """
    Generates high-impact scalability line plots across grid sizes.
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_premium_style()
    
    df_plot = df_global.copy()
    df_plot['GridNodes'] = df_plot['Case'].str.extract('(\d+)').astype(int)
    df_plot = df_plot[df_plot['Success'] == True]
    
    hue_order = ["Generic Flat Start (1.0 p.u., 0°)", "Linearized DC Start", "Physics-Informed GNN"]
    palette = {
        "Generic Flat Start (1.0 p.u., 0°)": THEME_COLORS["palette"][0],
        "Linearized DC Start": THEME_COLORS["palette"][1],
        "Physics-Informed GNN": THEME_COLORS["palette"][2]
    }

    plt.figure(figsize=(9, 6))
    sns.lineplot(data=df_plot, x="GridNodes", y="Time_ms", hue="InitMethod", 
                 hue_order=hue_order, palette=palette, style="InitMethod", 
                 markers=True, markersize=12, linewidth=3, errorbar='se')
                 
    plt.title("Benchmarking Scalability: Computational Cost vs. Network Nodes", weight='bold', pad=20)
    plt.xlabel("Graph Dimension (Nodes)")
    plt.ylabel("Solve Time (ms)")
    plt.yscale('log')
    
    plt.xticks(sorted(df_plot['GridNodes'].unique()))
    plt.legend(title="", loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scaling_speed_premium.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[AESTHETICS] Premium warm-start benchmarks generated to {output_dir}")
