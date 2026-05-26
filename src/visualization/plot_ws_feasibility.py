from typing import List, Dict
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from src.visualization.plot_ws_common import apply_ws_style, METHOD_COLORS, save_pub_figure

MODEL_PALETTE = {
    "StandardGCN": "#E63946",       # Vibrant Red
    "DynamicGCN": "#457B9D",        # Steel Blue
    "PIGCN": "#2A9D8F",             # Teal
    "PIGCLSTM": "#F4A261",          # Sandy Orange
    "PIGCGRU": "#9B5DE5",           # Purple
    "PIResnetGCLSTM": "#F15BB5",     # Pink
    "PIResnetGCGRU": "#00F5D4",      # Neon Turquoise
    "flat": "#ADB5BD",              # Neutral Gray
    "dc": "#6C757D"                 # Dark Gray
}

def plot_ws_feasibility(records: List[Dict], summary: Dict[str, float], case_name: str, output_dir: str) -> None:
    """
    Generate highly professional, publication-quality plots for feasibility.
    Includes sequential constraint satisfaction line trajectories and feasibility rate summaries.
    """
    if not records:
        return

    apply_ws_style()
    df = pd.DataFrame(records)

    # Model name resolution for legends
    model_name = summary.get("model_name", "Neural Model")
    if model_name.lower() == "none":
        model_name = "Neural Warmstart"

    # Color registry alignment
    colors = {
        "flat": METHOD_COLORS.get("flat", "#ADB5BD"),
        "dc": METHOD_COLORS.get("dc", "#6C757D"),
        "warmstart": METHOD_COLORS.get("warmstart", "#E63946")
    }

    # Label improvements
    flat_label = "Generic Flat Start (1.0 p.u., 0°)"
    dc_label = "Linearized DC Start"
    model_label = f"Physics-Informed GNN ({model_name})"

    # 1) Feasibility Success Rate Summary (Bar Chart)
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ["flat", "dc", "warmstart"]
    rates = [100.0 * summary.get(f"{m}_feasibility_rate", 0.0) for m in methods]
    labels = [flat_label, dc_label, model_label]
    bars = ax.bar(labels, rates, color=[colors["flat"], colors["dc"], colors["warmstart"]], width=0.4)
    ax.set_ylim(0, 110)
    ax.set_title(f"{case_name.upper()} - Physical Solution Feasibility Rate", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Feasible Solutions (%)", fontsize=10, fontweight="bold")
    ax.set_xticklabels(labels, fontsize=8)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=9)
    sns.despine(ax=ax)
    save_pub_figure(fig, output_dir, f"{case_name}_feasibility_rate")
    plt.close(fig)

    # Pivot records to compile sample-by-sample comparisons for line plots
    try:
        pivot_df = df.pivot(index="sample_id", columns="method", values="constraint_satisfaction_rate")
        # Reset index to treat sample_id as a regular column
        pivot_df = pivot_df.reset_index()
        
        # Sort samples by Flat Start feasibility to show comparative trends
        pivot_df = pivot_df.sort_values(by="flat").reset_index(drop=True)
        x_indices = np.arange(len(pivot_df))

        # 2) Constraint Satisfaction Trajectory (Line Plot with vertical offset for visibility)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        
        # Displace flat by -0.5% and warmstart by +0.5% so overlapping 100% lines are fully visible
        ax.plot(x_indices, pivot_df["flat"] * 100.0 - 0.5, color=colors["flat"], label=flat_label, linestyle=":", marker="o", markersize=4, alpha=0.8)
        ax.plot(x_indices, pivot_df["dc"] * 100.0, color=colors["dc"], label=dc_label, linestyle="--", marker="s", markersize=4, alpha=0.8)
        ax.plot(x_indices, pivot_df["warmstart"] * 100.0 + 0.5, color=colors["warmstart"], label=model_label, linestyle="-", marker="^", markersize=5, linewidth=1.8)
        
        ax.set_ylim(-5, 105)
        ax.set_title(f"{case_name.upper()} - Constraint Satisfaction Rate per State (Offset for Visibility)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Operational State Sample Index (Sorted by Flat Start Rate)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Physical Constraints Satisfied (%, Offset ±0.5%)", fontsize=10, fontweight="bold")
        ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        sns.despine(ax=ax)
        save_pub_figure(fig, output_dir, f"{case_name}_constraint_satisfaction_trajectory")
        plt.close(fig)

    except Exception as e:
        print(f"[WARNING] Feasibility trajectory line plotting skipped: {e}")


def plot_ws_feasibility_comparison(df_all: pd.DataFrame, case_name: str, output_dir: str) -> None:
    """
    Generate grouped comparison bar charts for feasibility rates and constraint satisfaction rates
    across multiple GNN models.
    """
    os.makedirs(output_dir, exist_ok=True)
    apply_ws_style()

    # Map to concise professional labels
    METHOD_MAP = {
        "flat": "Generic Flat Start (1.0 p.u., 0°)",
        "dc": "Linearized DC Start",
        "warmstart": "Physics-Informed GNN"
    }
    
    df_plot = df_all.copy()
    df_plot["InitMethod"] = df_plot["method"].map(METHOD_MAP)

    hue_order = ["Generic Flat Start (1.0 p.u., 0°)", "Linearized DC Start", "Physics-Informed GNN"]
    palette = {
        "Generic Flat Start (1.0 p.u., 0°)": MODEL_PALETTE["flat"],
        "Linearized DC Start": MODEL_PALETTE["dc"],
        "Physics-Informed GNN": MODEL_PALETTE["StandardGCN"] # Standard GNN color for comparisons
    }

    # 1) Feasibility Success Rate Comparison (Grouped Bar Chart)
    plt.figure(figsize=(11, 5.5))
    feasibility_df = df_plot.groupby(["model", "InitMethod"])["is_feasible"].mean().reset_index()
    feasibility_df["is_feasible"] *= 100.0  # Convert to %

    ax1 = sns.barplot(
        data=feasibility_df,
        x="model",
        y="is_feasible",
        hue="InitMethod",
        hue_order=hue_order,
        palette=palette,
        edgecolor="white",
        linewidth=1.2
    )

    plt.title(f"[{case_name.upper()}] Physical Feasibility Comparison across Models", weight="bold", pad=15)
    plt.ylabel("Feasible Solutions (%)", fontweight="bold")
    plt.xlabel("Model Architecture", fontweight="bold")
    plt.ylim(0, 110)
    plt.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    save_pub_figure(plt.gcf(), output_dir, f"{case_name}_all_models_feasibility_rate_comparison")
    plt.close()

    # 2) Constraint Satisfaction Rate Comparison (Grouped Bar Chart)
    plt.figure(figsize=(11, 5.5))
    csr_df = df_plot.groupby(["model", "InitMethod"])["constraint_satisfaction_rate"].mean().reset_index()
    csr_df["constraint_satisfaction_rate"] *= 100.0  # Convert to %

    ax2 = sns.barplot(
        data=csr_df,
        x="model",
        y="constraint_satisfaction_rate",
        hue="InitMethod",
        hue_order=hue_order,
        palette=palette,
        edgecolor="white",
        linewidth=1.2
    )

    plt.title(f"[{case_name.upper()}] Constraint Satisfaction Rate Comparison across Models", weight="bold", pad=15)
    plt.ylabel("Constraints Satisfied (%)", fontweight="bold")
    plt.xlabel("Model Architecture", fontweight="bold")
    plt.ylim(0, 110)
    plt.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    save_pub_figure(plt.gcf(), output_dir, f"{case_name}_all_models_constraint_satisfaction_comparison")
    plt.close()
