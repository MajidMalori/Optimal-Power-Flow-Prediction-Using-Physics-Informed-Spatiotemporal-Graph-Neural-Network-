import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.visualization.plot_uncertainty import load_network_topology


def _apply_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
        }
    )


def plot_pred_vs_true_scatter(
    vm_pred: np.ndarray,
    va_pred: np.ndarray,
    vm_true: np.ndarray,
    va_true: np.ndarray,
    output_path: str,
    model_name: str,
    case_name: str,
):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    vm_pred_flat = vm_pred.reshape(-1)
    vm_true_flat = vm_true.reshape(-1)
    va_pred_flat = va_pred.reshape(-1)
    va_true_flat = va_true.reshape(-1)

    axes[0].scatter(vm_true_flat, vm_pred_flat, s=8, alpha=0.35, color="#1f77b4")
    lims = [
        min(vm_true_flat.min(), vm_pred_flat.min()),
        max(vm_true_flat.max(), vm_pred_flat.max()),
    ]
    axes[0].plot(lims, lims, "--", color="black", linewidth=1)
    axes[0].set_xlabel("True $V_m$ (p.u.)")
    axes[0].set_ylabel("Predicted $V_m$ (p.u.)")
    axes[0].set_title("Voltage Magnitude")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(va_true_flat, va_pred_flat, s=8, alpha=0.35, color="#d62728")
    lims = [
        min(va_true_flat.min(), va_pred_flat.min()),
        max(va_true_flat.max(), va_pred_flat.max()),
    ]
    axes[1].plot(lims, lims, "--", color="black", linewidth=1)
    axes[1].set_xlabel("True $\\theta$ (rad)")
    axes[1].set_ylabel("Predicted $\\theta$ (rad)")
    axes[1].set_title("Voltage Angle")
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"{model_name} on {case_name}: Predicted vs True", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_network_voltage_panel(
    case_name: str,
    vm_values: np.ndarray,
    va_values: np.ndarray,
    output_path: str,
    title: str,
    value_key: str = "vm",
):
    _apply_style()
    _, graph, pos = load_network_topology(case_name)
    values = vm_values if value_key == "vm" else va_values
    label = "$V_m$ (p.u.)" if value_key == "vm" else "$\\theta$ (rad)"
    cmap = "viridis" if value_key == "vm" else "coolwarm"

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25, width=1.2, edge_color="gray")
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=values,
        node_size=220 if case_name != "case118" else 120,
        cmap=cmap,
        edgecolors="black",
        linewidths=0.6,
    )
    if case_name != "case118":
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7)

    cbar = fig.colorbar(nodes, ax=ax, shrink=0.85)
    cbar.set_label(label)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_sample_network_comparisons(
    case_name: str,
    sample_records: Sequence[Dict],
    output_dir: str,
    model_name: str,
):
    os.makedirs(output_dir, exist_ok=True)
    for record in sample_records:
        sid = record["sample_id"]
        plot_network_voltage_panel(
            case_name,
            np.asarray(record["vm_pred"]),
            np.asarray(record["va_pred"]),
            os.path.join(output_dir, f"{sid}_vm_pred.png"),
            title=f"{sid}\nPredicted $V_m$",
            value_key="vm",
        )
        plot_network_voltage_panel(
            case_name,
            np.asarray(record["vm_true"]),
            np.asarray(record["va_true"]),
            os.path.join(output_dir, f"{sid}_vm_true.png"),
            title=f"{sid}\nTrue $V_m$",
            value_key="vm",
        )
        plot_network_voltage_panel(
            case_name,
            np.asarray(record["vm_pred"]),
            np.asarray(record["va_pred"]),
            os.path.join(output_dir, f"{sid}_va_pred.png"),
            title=f"{sid}\nPredicted $\\theta$",
            value_key="va",
        )


def plot_fraction_temporal_mae(
    metadata: Sequence[Dict],
    vm_pred: np.ndarray,
    va_pred: np.ndarray,
    vm_true: np.ndarray,
    va_true: np.ndarray,
    output_path: str,
    model_name: str,
    case_name: str,
):
    _apply_style()
    fractions = sorted({m["renewable_fraction"] for m in metadata})
    vm_mae_by_frac: List[List[float]] = []
    va_mae_by_frac: List[List[float]] = []

    for frac in fractions:
        idxs = [i for i, m in enumerate(metadata) if m["renewable_fraction"] == frac]
        timesteps = sorted({metadata[i]["timestep"] for i in idxs})
        vm_curve, va_curve = [], []
        for t in timesteps:
            t_idxs = [i for i in idxs if metadata[i]["timestep"] == t]
            vm_curve.append(float(np.mean(np.abs(vm_pred[t_idxs] - vm_true[t_idxs]))))
            va_curve.append(float(np.mean(np.abs(va_pred[t_idxs] - va_true[t_idxs]))))
        vm_mae_by_frac.append(vm_curve)
        va_mae_by_frac.append(va_curve)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for frac, vm_curve, va_curve in zip(fractions, vm_mae_by_frac, va_mae_by_frac):
        label = f"{int(round(frac * 100))}% ren."
        axes[0].plot(vm_curve, marker="o", label=label)
        axes[1].plot(va_curve, marker="o", label=label)

    axes[0].set_title("Mean $|V_m^{pred} - V_m^{true}|$ by timestep")
    axes[1].set_title("Mean $|\\theta^{pred} - \\theta^{true}|$ by timestep")
    for ax in axes:
        ax.set_xlabel("Test timestep index within fraction block")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"{model_name} on {case_name}: temporal error by renewable fraction", fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_sample_network_comparison_panel(
    case_name: str,
    record: Dict,
    output_path: str,
    model_name: str,
):
    _apply_style()
    _, graph, pos = load_network_topology(case_name)

    vm_pred = np.asarray(record["vm_pred"])
    vm_true = np.asarray(record["vm_true"])
    va_pred = np.asarray(record["va_pred"])
    va_true = np.asarray(record["va_true"])
    sid = record["sample_id"]

    vm_min = min(vm_pred.min(), vm_true.min())
    vm_max = max(vm_pred.max(), vm_true.max())
    va_min = min(va_pred.min(), va_true.min())
    va_max = max(va_pred.max(), va_true.max())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    node_size = 220 if case_name != "case118" else 120
    font_size = 7

    def draw_panel(ax, values, title, cmap, vmin, vmax, label):
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25, width=1.2, edge_color="gray")
        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=values,
            node_size=node_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
            linewidths=0.6,
        )
        if case_name != "case118":
            nx.draw_networkx_labels(graph, pos, ax=ax, font_size=font_size)
        cbar = fig.colorbar(nodes, ax=ax, shrink=0.75)
        cbar.set_label(label)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    draw_panel(axes[0, 0], vm_pred, "Predicted $V_m$", "viridis", vm_min, vm_max, "$V_m$ (p.u.)")
    draw_panel(axes[0, 1], vm_true, "True $V_m$", "viridis", vm_min, vm_max, "$V_m$ (p.u.)")
    draw_panel(axes[1, 0], va_pred, "Predicted $\\theta$", "coolwarm", va_min, va_max, "$\\theta$ (rad)")
    draw_panel(axes[1, 1], va_true, "True $\\theta$", "coolwarm", va_min, va_max, "$\\theta$ (rad)")

    fig.suptitle(f"{model_name} on {case_name} (Sample: {sid})\nGrid State Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_combined_pred_vs_true_scatter(
    model_data: List[Dict],
    output_path: str,
    case_name: str,
):
    _apply_style()
    n_models = len(model_data)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 4.5 * n_models), squeeze=False)

    for i, data in enumerate(model_data):
        m_name = data["model_name"]
        vm_pred_flat = data["vm_pred"].reshape(-1)
        vm_true_flat = data["vm_true"].reshape(-1)
        va_pred_flat = data["va_pred"].reshape(-1)
        va_true_flat = data["va_true"].reshape(-1)

        # Calculate Pearson correlation coefficient
        try:
            r_vm = np.corrcoef(vm_true_flat, vm_pred_flat)[0, 1]
            if np.isnan(r_vm):
                r_vm = 0.0
        except Exception:
            r_vm = 0.0

        try:
            r_va = np.corrcoef(va_true_flat, va_pred_flat)[0, 1]
            if np.isnan(r_va):
                r_va = 0.0
        except Exception:
            r_va = 0.0

        axes[i, 0].scatter(vm_true_flat, vm_pred_flat, s=8, alpha=0.35, color="#1f77b4")
        lims = [
            min(vm_true_flat.min(), vm_pred_flat.min()),
            max(vm_true_flat.max(), vm_pred_flat.max()),
        ]
        axes[i, 0].plot(lims, lims, "--", color="black", linewidth=1)
        axes[i, 0].set_xlabel("True $V_m$ (p.u.)")
        axes[i, 0].set_ylabel(f"{m_name}\nPredicted $V_m$ (p.u.)")
        axes[i, 0].grid(alpha=0.25)
        
        # Add correlation value text box
        axes[i, 0].text(
            0.05,
            0.95,
            f"$r$ = {r_vm:.4f}",
            transform=axes[i, 0].transAxes,
            verticalalignment="top",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="lightgray")
        )
        
        if i == 0:
            axes[i, 0].set_title("Voltage Magnitude", fontsize=12, fontweight="bold")

        axes[i, 1].scatter(va_true_flat, va_pred_flat, s=8, alpha=0.35, color="#d62728")
        lims = [
            min(va_true_flat.min(), va_pred_flat.min()),
            max(va_true_flat.max(), va_pred_flat.max()),
        ]
        axes[i, 1].plot(lims, lims, "--", color="black", linewidth=1)
        axes[i, 1].set_xlabel("True $\\theta$ (rad)")
        axes[i, 1].set_ylabel("Predicted $\\theta$ (rad)")
        axes[i, 1].grid(alpha=0.25)
        
        # Add correlation value text box
        axes[i, 1].text(
            0.05,
            0.95,
            f"$r$ = {r_va:.4f}",
            transform=axes[i, 1].transAxes,
            verticalalignment="top",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="lightgray")
        )
        
        if i == 0:
            axes[i, 1].set_title("Voltage Angle", fontsize=12, fontweight="bold")

    fig.suptitle(f"Predicted vs True Scatter Plots on {case_name}", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_combined_fraction_temporal_mae(
    model_data: List[Dict],
    output_path: str,
    case_name: str,
):
    _apply_style()
    n_models = len(model_data)
    fig, axes = plt.subplots(n_models, 2, figsize=(13, 4.5 * n_models), squeeze=False)

    for idx, data in enumerate(model_data):
        m_name = data["model_name"]
        metadata = data["metadata"]
        vm_pred = data["vm_pred"]
        va_pred = data["va_pred"]
        vm_true = data["vm_true"]
        va_true = data["va_true"]

        fractions = sorted({m["renewable_fraction"] for m in metadata})
        vm_mae_by_frac: List[List[float]] = []
        va_mae_by_frac: List[List[float]] = []

        for frac in fractions:
            idxs = [i for i, m in enumerate(metadata) if m["renewable_fraction"] == frac]
            timesteps = sorted({metadata[i]["timestep"] for i in idxs})
            vm_curve, va_curve = [], []
            for t in timesteps:
                t_idxs = [i for i in idxs if metadata[i]["timestep"] == t]
                vm_curve.append(float(np.mean(np.abs(vm_pred[t_idxs] - vm_true[t_idxs]))))
                va_curve.append(float(np.mean(np.abs(va_pred[t_idxs] - va_true[t_idxs]))))
            vm_mae_by_frac.append(vm_curve)
            va_mae_by_frac.append(va_curve)

        for frac, vm_curve, va_curve in zip(fractions, vm_mae_by_frac, va_mae_by_frac):
            label = f"{int(round(frac * 100))}% ren."
            axes[idx, 0].plot(vm_curve, marker="o", label=label)
            axes[idx, 1].plot(va_curve, marker="o", label=label)

        axes[idx, 0].set_ylabel(f"{m_name}\nMean $|V_m^{{pred}} - V_m^{{true}}|$")
        axes[idx, 1].set_ylabel("Mean $|\\theta^{pred} - \\theta^{true}|$")

        for col in [0, 1]:
            axes[idx, col].set_xlabel("Test timestep index within fraction block")
            axes[idx, col].grid(alpha=0.25)
            axes[idx, col].legend(fontsize=8)

        if idx == 0:
            axes[idx, 0].set_title("Voltage Magnitude MAE", fontsize=12, fontweight="bold")
            axes[idx, 1].set_title("Voltage Angle MAE", fontsize=12, fontweight="bold")

    fig.suptitle(f"Temporal Error by Renewable Fraction on {case_name}", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
