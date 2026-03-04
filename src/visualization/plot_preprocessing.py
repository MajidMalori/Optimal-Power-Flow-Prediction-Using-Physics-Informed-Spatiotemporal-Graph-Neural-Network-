"""
Preprocessing Diagnostic Visualization
Generates 4 individual professional plots to validate the preprocessed data:
1. Feature Distribution (Box Plots) - Scale consistency check
2. Feature Correlation Heatmap - Redundancy/leakage detection
3. Train/Val/Test Distribution Comparison - Distribution shift detection
4. Temporal Continuity - Chronological split sanity check
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.constants import FeatureIndices

FEATURE_NAMES = FeatureIndices.FEATURE_NAMES

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
})


def _load_splits(case_dir):
    train_f = torch.load(os.path.join(case_dir, 'train_features.pt'), weights_only=True).numpy()
    val_f = torch.load(os.path.join(case_dir, 'val_features.pt'), weights_only=True).numpy()
    test_f = torch.load(os.path.join(case_dir, 'test_features.pt'), weights_only=True).numpy()
    return train_f, val_f, test_f


def plot_feature_distribution(case_dir: str, case_name: str, output_path: str) -> str:
    """Box plots of all normalized features to verify scale consistency.
    Continuous features and structural degree are on separate subplots.
    Excludes unobserved/masked buses (vm == -1.0) for a clean representation."""
    train_f, val_f, test_f = _load_splits(case_dir)
    n_features = train_f.shape[2]
    names = FEATURE_NAMES[:n_features]

    all_data = np.concatenate([train_f, val_f, test_f], axis=0)
    flat = all_data.reshape(-1, n_features)

    vm_idx = FeatureIndices.VM
    has_degree = n_features > 10
    cont_count = n_features - 1 if has_degree else n_features

    # Build box data for continuous features, filtering masked vm buses
    cont_data = []
    cont_names = []
    for i in range(cont_count):
        col = flat[:, i]
        if i == vm_idx:
            col = col[col > -0.9]  # exclude unobserved buses
        cont_data.append(col)
        cont_names.append(names[i])

    if has_degree:
        fig, (ax_cont, ax_deg) = plt.subplots(1, 2, figsize=(15, 6),
                                               gridspec_kw={'width_ratios': [5, 1]})
    else:
        fig, ax_cont = plt.subplots(figsize=(14, 6))
        ax_deg = None

    # Continuous features box plot
    bp = ax_cont.boxplot(cont_data, labels=cont_names, patch_artist=True, showfliers=False,
                         medianprops=dict(color='black', linewidth=1.5))
    colors = plt.cm.Set3(np.linspace(0, 1, cont_count))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax_cont.set_title('Continuous Features', fontsize=12)
    ax_cont.set_ylabel('Per-Unit Value')
    ax_cont.tick_params(axis='x', rotation=45)
    ax_cont.grid(axis='y', alpha=0.3, linestyle='--')
    ax_cont.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # Annotate vm to clarify it's centered
    vm_pos = vm_idx + 1  # boxplot positions are 1-indexed
    ax_cont.annotate('centered\naround 1.0', xy=(vm_pos, 0), xytext=(vm_pos + 0.8, 0.06),
                     fontsize=8, color='#555555', style='italic',
                     arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))

    # Structural degree feature (separate scale)
    if has_degree and ax_deg is not None:
        deg_col = flat[:, FeatureIndices.DEGREE]
        bp_deg = ax_deg.boxplot([deg_col], labels=['degree'], patch_artist=True, showfliers=False,
                                medianprops=dict(color='black', linewidth=1.5))
        bp_deg['boxes'][0].set_facecolor('#aec6cf')
        bp_deg['boxes'][0].set_alpha(0.8)
        ax_deg.set_title('Structural', fontsize=12)
        ax_deg.set_ylabel('Normalized Degree')
        ax_deg.set_ylim(0, 1.1)
        ax_deg.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle(f'Feature Scale Distribution (Post-Normalization) — {case_name.upper()}',
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


def plot_correlation_heatmap(case_dir: str, case_name: str, output_path: str) -> str:
    """Correlation heatmap from training data to detect redundancy or leakage."""
    train_f, _, _ = _load_splits(case_dir)
    n_features = train_f.shape[2]
    names = FEATURE_NAMES[:n_features]

    train_flat = train_f.reshape(-1, n_features)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(train_flat.T)
    corr = np.nan_to_num(corr)  # Handle constant features (NaN correlation)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                vmin=-1, vmax=1, center=0, square=True,
                xticklabels=names, yticklabels=names,
                ax=ax, cbar_kws={'shrink': 0.8}, annot_kws={'size': 9})
    ax.set_title(f'Feature Correlation Matrix (Training Set) — {case_name.upper()}', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


def plot_split_distribution(case_dir: str, case_name: str, output_path: str) -> str:
    """Bar chart comparing mean±std of key features across train/val/test.
    Continuous features and structural (degree) are plotted on separate subplots
    for clear scale comparison. Masked buses are excluded from vm statistics."""
    train_f, val_f, test_f = _load_splits(case_dir)
    n_features = train_f.shape[2]
    names = FEATURE_NAMES[:n_features]

    continuous_indices = [FeatureIndices.P_LOAD, FeatureIndices.P_REN, FeatureIndices.VM]
    has_degree = n_features > 10

    splits = {'Train': train_f, 'Val': val_f, 'Test': test_f}
    split_colors = {'Train': '#2ecc71', 'Val': '#3498db', 'Test': '#e74c3c'}
    width = 0.25
    vm_idx = FeatureIndices.VM

    if has_degree:
        fig, (ax_cont, ax_deg) = plt.subplots(1, 2, figsize=(13, 6),
                                               gridspec_kw={'width_ratios': [3, 1]})
    else:
        fig, ax_cont = plt.subplots(figsize=(10, 6))
        ax_deg = None

    # Continuous features subplot
    cont_names = [names[i] for i in continuous_indices]
    x_cont = np.arange(len(continuous_indices))
    for j, (split_name, data) in enumerate(splits.items()):
        flat_split = data.reshape(-1, n_features)
        means, stds = [], []
        for i in continuous_indices:
            col = flat_split[:, i]
            if i == vm_idx:
                col = col[col > -0.9]  # exclude masked buses
            means.append(col.mean())
            stds.append(col.std())
        ax_cont.bar(x_cont + j * width, means, width, yerr=stds, label=split_name,
                    color=split_colors[split_name], alpha=0.8, capsize=3)

    ax_cont.set_xticks(x_cont + width)
    ax_cont.set_xticklabels(cont_names)
    ax_cont.set_ylabel('Per-Unit Value')
    ax_cont.set_title('Continuous Features', fontsize=12)
    ax_cont.legend(frameon=True, shadow=True)
    ax_cont.grid(axis='y', alpha=0.3, linestyle='--')

    # Degree subplot (structural feature, different scale)
    if has_degree and ax_deg is not None:
        deg_idx = FeatureIndices.DEGREE
        x_deg = np.arange(1)
        for j, (split_name, data) in enumerate(splits.items()):
            flat_split = data.reshape(-1, n_features)
            deg_col = flat_split[:, deg_idx]
            ax_deg.bar(x_deg + j * width, [deg_col.mean()], width,
                       yerr=[deg_col.std()], label=split_name,
                       color=split_colors[split_name], alpha=0.8, capsize=3)
        ax_deg.set_xticks(x_deg + width)
        ax_deg.set_xticklabels(['degree'])
        ax_deg.set_ylabel('Normalized Degree')
        ax_deg.set_title('Structural Feature', fontsize=12)
        ax_deg.set_ylim(0, 1.1)
        ax_deg.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle(f'Distribution Shift Check (Mean ± Std) — {case_name.upper()}',
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


def plot_temporal_continuity(case_dir: str, case_name: str, output_path: str) -> str:
    """Time-series plot of key features across the full chronological sequence with split boundaries."""
    train_f, val_f, test_f = _load_splits(case_dir)

    full_sequence = np.concatenate([train_f, val_f, test_f], axis=0)
    t_total = full_sequence.shape[0]
    t_train = train_f.shape[0]
    t_val = val_f.shape[0]

    # Compute means excluding masked buses (vm == -1.0) for cleaner visualization
    vm_data = full_sequence[:, :, FeatureIndices.VM]
    vm_masked = np.ma.masked_where(vm_data < -0.9, vm_data)
    vm_mean = vm_masked.mean(axis=1)
    p_load_mean = full_sequence[:, :, FeatureIndices.P_LOAD].mean(axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(t_total), vm_mean, color='#2980b9', alpha=0.8, linewidth=1, label='Mean V_m deviation (p.u.)')
    ax_twin = ax.twinx()
    ax_twin.plot(range(t_total), p_load_mean, color='#e67e22', alpha=0.8, linewidth=1, label='Mean P_Load (p.u.)')

    ax.axvline(x=t_train, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Train → Val')
    ax.axvline(x=t_train + t_val, color='darkred', linestyle='--', linewidth=2, alpha=0.8, label='Val → Test')

    ax.axvspan(0, t_train, alpha=0.05, color='green')
    ax.axvspan(t_train, t_train + t_val, alpha=0.05, color='blue')
    ax.axvspan(t_train + t_val, t_total, alpha=0.05, color='red')

    ax.set_xlabel('Timestep (Chronological)')
    ax.set_ylabel('Mean V_m Deviation from 1.0 (p.u.)', color='#2980b9')
    ax_twin.set_ylabel('Mean P_Load (p.u.)', color='#e67e22')
    ax.set_title(f'Temporal Continuity Across Splits — {case_name.upper()}', fontweight='bold', pad=25)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    # Move legend outside to the right
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
              bbox_to_anchor=(1.08, 1), frameon=True, shadow=True)

    # Position labels at the very top of the graph using axes coordinates (0-1)
    # y=1.01 is just above the top spine
    ax.text((t_train/2)/t_total, 1.01, 'TRAIN', transform=ax.transAxes, 
            ha='center', fontsize=10, color='green', fontweight='bold')
    ax.text((t_train + t_val/2)/t_total, 1.01, 'VAL', transform=ax.transAxes, 
            ha='center', fontsize=10, color='blue', fontweight='bold')
    ax.text((t_train + t_val + (t_total - t_train - t_val) / 2)/t_total, 1.01, 'TEST', 
            transform=ax.transAxes, ha='center', fontsize=10, color='red', fontweight='bold')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return output_path


def generate_all_preprocessing_plots(case_dir: str, case_name: str, reports_dir: str) -> list:
    """Generate all 4 preprocessing diagnostic plots as separate PNGs."""
    from tqdm import tqdm
    print()

    num_buses = case_name.replace('case', '')
    os.makedirs(reports_dir, exist_ok=True)

    plots = [
        ('feature_distribution', plot_feature_distribution),
        ('correlation_heatmap', plot_correlation_heatmap),
        ('split_distribution', plot_split_distribution),
        ('temporal_continuity', plot_temporal_continuity),
    ]

    paths = []
    pbar = tqdm(plots, desc=f"Plotting {case_name}", unit="plot",
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n}/{total} plots",
                leave=True)
    for name, fn in pbar:
        # Update description with current plot name, keeping it concise
        pbar.set_description(f"Plotting {case_name} ({name})")
        output_path = os.path.join(reports_dir, f'{name}_{num_buses}bus.png')
        result = fn(case_dir, case_name, output_path)
        if result:
            paths.append(result)
    return paths
