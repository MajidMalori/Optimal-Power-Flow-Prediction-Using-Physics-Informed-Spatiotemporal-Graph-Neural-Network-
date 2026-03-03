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

from constants import FeatureIndices

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
    """Box plots of all normalized features to verify scale consistency."""
    train_f, val_f, test_f = _load_splits(case_dir)
    n_features = train_f.shape[2]
    names = FEATURE_NAMES[:n_features]

    all_data = np.concatenate([train_f, val_f, test_f], axis=0)
    flat = all_data.reshape(-1, n_features)

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot([flat[:, i] for i in range(n_features)],
                    labels=names, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5))
    colors = plt.cm.Set3(np.linspace(0, 1, n_features))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title(f'Feature Scale Distribution (Post-Normalization) — {case_name.upper()}', fontweight='bold')
    ax.set_ylabel('Per-Unit Value')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

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
    """Bar chart comparing mean±std of key features across train/val/test."""
    train_f, val_f, test_f = _load_splits(case_dir)
    n_features = train_f.shape[2]
    names = FEATURE_NAMES[:n_features]

    key_indices = [FeatureIndices.P_LOAD, FeatureIndices.P_REN, FeatureIndices.VM]
    if n_features > 10:
        key_indices.append(FeatureIndices.DEGREE)
    key_names = [names[i] for i in key_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(key_indices))
    width = 0.25
    splits = {'Train': train_f, 'Val': val_f, 'Test': test_f}
    split_colors = {'Train': '#2ecc71', 'Val': '#3498db', 'Test': '#e74c3c'}

    for j, (split_name, data) in enumerate(splits.items()):
        flat_split = data.reshape(-1, n_features)
        means = [flat_split[:, i].mean() for i in key_indices]
        stds = [flat_split[:, i].std() for i in key_indices]
        ax.bar(x + j * width, means, width, yerr=stds, label=split_name,
               color=split_colors[split_name], alpha=0.8, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(key_names, rotation=45)
    ax.set_title(f'Distribution Shift Check (Mean ± Std) — {case_name.upper()}', fontweight='bold')
    ax.set_ylabel('Per-Unit Value')
    ax.legend(frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

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

    vm_mean = full_sequence[:, :, FeatureIndices.VM].mean(axis=1)
    p_load_mean = full_sequence[:, :, FeatureIndices.P_LOAD].mean(axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(t_total), vm_mean, color='#2980b9', alpha=0.8, linewidth=1, label='Mean V_m (p.u.)')
    ax_twin = ax.twinx()
    ax_twin.plot(range(t_total), p_load_mean, color='#e67e22', alpha=0.8, linewidth=1, label='Mean P_Load (p.u.)')

    ax.axvline(x=t_train, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Train → Val')
    ax.axvline(x=t_train + t_val, color='darkred', linestyle='--', linewidth=2, alpha=0.8, label='Val → Test')

    ax.axvspan(0, t_train, alpha=0.05, color='green')
    ax.axvspan(t_train, t_train + t_val, alpha=0.05, color='blue')
    ax.axvspan(t_train + t_val, t_total, alpha=0.05, color='red')

    ax.set_xlabel('Timestep (Chronological)')
    ax.set_ylabel('Mean Voltage (p.u.)', color='#2980b9')
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
