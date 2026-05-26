from typing import List, Dict
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from src.visualization.plot_ws_common import apply_ws_style, METHOD_COLORS, save_pub_figure


def plot_ws_speed(records: List[Dict], summary: Dict[str, float], case_name: str, output_dir: str) -> None:
    """
    Generate highly professional, publication-quality plots for warm-start speed.
    Includes both time distributions and sample-by-sample trajectory line plots.
    """
    if not records:
        return

    apply_ws_style()
    df = pd.DataFrame(records)
    df_ok = df[df["converged"] == True]

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

    # 1) Solve Time Distribution (Log Scale Boxplot)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.boxplot(
        data=df_ok,
        x="method",
        y="time_ms",
        order=["flat", "dc", "warmstart"],
        hue="method",
        legend=False,
        palette=colors,
        width=0.5,
        ax=ax,
        fliersize=3
    )
    ax.set_yscale("log")
    ax.set_title(f"{case_name.upper()} - Power Flow Solve Time Distribution (Log Scale)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Initialization Method", fontsize=10, fontweight="bold")
    ax.set_ylabel("Execution Time (ms)", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([flat_label, dc_label, model_label], fontsize=8)
    sns.despine(ax=ax)
    save_pub_figure(fig, output_dir, f"{case_name}_speed_time_distribution")
    plt.close(fig)

    # Pivot records to compile sample-by-sample comparisons for line plots
    try:
        pivot_df = df.pivot(index="sample_id", columns="method", values=["time_ms", "iterations", "converged"])
        # Reset index to treat sample_id as a regular column
        pivot_df = pivot_df.reset_index()
        
        # Flatten multi-level column names
        pivot_df.columns = [
            f"{col[1]}_{col[0]}" if col[1] else col[0] 
            for col in pivot_df.columns
        ]
        
        # Sort samples by Flat Start solve time to form a smooth scaling curve
        pivot_df = pivot_df.sort_values(by="flat_time_ms").reset_index(drop=True)
        x_indices = np.arange(len(pivot_df))

        # 2) Sample-by-Sample Solve Time Trajectory (Line Plot)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(x_indices, pivot_df["flat_time_ms"], color=colors["flat"], label=flat_label, linestyle=":", marker="o", markersize=4, alpha=0.8)
        ax.plot(x_indices, pivot_df["dc_time_ms"], color=colors["dc"], label=dc_label, linestyle="--", marker="s", markersize=4, alpha=0.8)
        ax.plot(x_indices, pivot_df["warmstart_time_ms"], color=colors["warmstart"], label=model_label, linestyle="-", marker="^", markersize=5, linewidth=1.8)
        
        ax.set_yscale("log")
        ax.set_title(f"{case_name.upper()} - Solver Solve Time comparison across states", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Operational State Sample Index (Sorted by Flat Start Time)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Solve Time (ms) - Log Scale", fontsize=10, fontweight="bold")
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        sns.despine(ax=ax)
        save_pub_figure(fig, output_dir, f"{case_name}_speed_time_trajectory")
        plt.close(fig)

        # 3) Sample-by-Sample Newton-Raphson Iterations (Line Plot with vertical displacement for visibility)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Displace vertically by +/- 0.05 so overlapping lines are fully visible
        ax.plot(x_indices, pivot_df["flat_iterations"] - 0.05, color=colors["flat"], label=flat_label, linestyle=":", marker="o", markersize=4, alpha=0.8)
        ax.plot(x_indices, pivot_df["dc_iterations"], color=colors["dc"], label=dc_label, linestyle="--", marker="s", markersize=4, alpha=0.8)
        ax.plot(x_indices, pivot_df["warmstart_iterations"] + 0.05, color=colors["warmstart"], label=model_label, linestyle="-", marker="^", markersize=5, linewidth=1.8)
        
        ax.set_title(f"{case_name.upper()} - Newton-Raphson Iterations per State (Offset for Visibility)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Operational State Sample Index (Sorted by Flat Start Time)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Solver Iterations (Vertical Offset ±0.05)", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
        
        max_iter_val = int(max(pivot_df["flat_iterations"].max(), pivot_df["dc_iterations"].max(), pivot_df["warmstart_iterations"].max()))
        ax.set_yticks(np.arange(0, max_iter_val + 2, 1))
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        sns.despine(ax=ax)
        save_pub_figure(fig, output_dir, f"{case_name}_speed_iterations_trajectory")
        plt.close(fig)

    except Exception as e:
        # Fallback if pivoting has duplicate or partial entries
        print(f"[WARNING] Trajectory line plotting skipped: {e}")

    # 4) Success and Speedup Summary Chart
    fig, ax = plt.subplots(figsize=(7, 5))
    rates = [
        100.0 * summary.get("flat_success_rate", 0.0),
        100.0 * summary.get("dc_success_rate", 0.0),
        100.0 * summary.get("warmstart_success_rate", 0.0),
    ]
    methods = [flat_label, dc_label, model_label]
    bars = ax.bar(methods, rates, color=[colors["flat"], colors["dc"], colors["warmstart"]], width=0.4)
    ax.set_ylim(0, 110)
    ax.set_title(f"{case_name.upper()} - Solver Convergence Success Rate", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Convergence Success Rate (%)", fontsize=10, fontweight="bold")
    ax.set_xticklabels(methods, fontsize=8)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=9)
    sns.despine(ax=ax)
    save_pub_figure(fig, output_dir, f"{case_name}_speed_success_rate")
    plt.close(fig)
