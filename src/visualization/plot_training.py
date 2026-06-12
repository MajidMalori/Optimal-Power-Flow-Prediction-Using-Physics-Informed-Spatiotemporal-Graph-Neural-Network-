import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: str) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _series(rows: List[Dict[str, str]], key: str) -> Optional[List[float]]:
    out: List[float] = []
    for r in rows:
        if key not in r or r[key] in (None, ""):
            return None
        out.append(float(r[key]))
    return out


def _epochs(rows: List[Dict[str, str]]) -> List[int]:
    return [int(float(r.get("epoch", i))) for i, r in enumerate(rows)]


def _style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_loss_curves(metrics_epoch_csv: str, output_path: str) -> Optional[str]:
    if not os.path.exists(metrics_epoch_csv):
        return None
    rows = _read_csv(metrics_epoch_csv)
    if not rows:
        return None

    _style()
    epochs = _epochs(rows)
    train_loss = _series(rows, "train_loss")
    val_loss = _series(rows, "val_loss")

    if train_loss is None and val_loss is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if train_loss is not None:
        ax.plot(epochs, train_loss, label="train_loss", linewidth=2)
    if val_loss is not None:
        ax.plot(epochs, val_loss, label="val_loss", linewidth=2)
        best_idx = int(min(range(len(val_loss)), key=lambda i: val_loss[i]))
        ax.axvline(epochs[best_idx], linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend(loc="best", frameon=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_lr_curve(metrics_epoch_csv: str, output_path: str) -> Optional[str]:
    if not os.path.exists(metrics_epoch_csv):
        return None
    rows = _read_csv(metrics_epoch_csv)
    if not rows:
        return None
    lr = _series(rows, "lr")
    if lr is None:
        return None

    _style()
    epochs = _epochs(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, lr, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning Rate Schedule")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_timing(metrics_epoch_csv: str, output_path: str) -> Optional[str]:
    if not os.path.exists(metrics_epoch_csv):
        return None
    rows = _read_csv(metrics_epoch_csv)
    if not rows:
        return None
    t = _series(rows, "epoch_time_s")
    if t is None:
        return None

    _style()
    epochs = _epochs(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, t, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_title("Epoch Wall Time")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_test_metrics(test_metrics_csv: str, output_path: str) -> Optional[str]:
    if not os.path.exists(test_metrics_csv):
        return None
    rows = _read_csv(test_metrics_csv)
    if not rows:
        return None
    r = rows[0]

    metrics = {k: float(v) for k, v in r.items() if k not in {"case", "model"} and v not in (None, "")}
    if not metrics:
        return None

    _style()
    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(keys)), vals)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Test Metrics (Scalar)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_case_loss_overlay(per_model_metrics_csvs: Dict[str, str], output_path: str) -> Optional[str]:
    series = []
    for model, path in per_model_metrics_csvs.items():
        if not os.path.exists(path):
            continue
        rows = _read_csv(path)
        if not rows:
            continue
        val_loss = _series(rows, "val_loss")
        if val_loss is None:
            continue
        series.append((model, _epochs(rows), val_loss))

    if not series:
        return None

    _style()
    fig, ax = plt.subplots(figsize=(11, 6))
    for model, epochs, val_loss in series:
        ax.plot(epochs, val_loss, linewidth=2, label=model)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val_loss")
    ax.set_title("Validation Loss Comparison")
    ax.legend(loc="best", frameon=True, ncol=2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_case_final_metrics(per_model_test_csvs: Dict[str, str], metric_key: str, output_path: str) -> Optional[str]:
    vals = []
    for model, path in per_model_test_csvs.items():
        if not os.path.exists(path):
            continue
        rows = _read_csv(path)
        if not rows:
            continue
        v = rows[0].get(metric_key, None)
        if v in (None, ""):
            continue
        vals.append((model, float(v)))

    if not vals:
        return None

    _style()
    models = [m for m, _ in vals]
    y = [v for _, v in vals]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(models)), y)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_key} Comparison")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_case_training_metrics_csv(per_model_test_csvs: Dict[str, str], output_path: str) -> Optional[str]:
    rows: List[Dict[str, str]] = []
    keys = set()
    for _model, path in per_model_test_csvs.items():
        if not os.path.exists(path):
            continue
        recs = _read_csv(path)
        if not recs:
            continue
        row = dict(recs[0])
        rows.append(row)
        keys.update(row.keys())

    if not rows:
        return None

    preferred = ["model", "case", "test_loss", "test_data_loss", "test_power_balance", "test_voltage_limit", "test_branch_capacity"]
    fieldnames = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    return output_path


def _median_epoch_time(metrics_csv_path: str) -> Optional[float]:
    if not os.path.exists(metrics_csv_path):
        return None
    rows = _read_csv(metrics_csv_path)
    vals = []
    for r in rows:
        v = _to_float(r.get("epoch_time_s"))
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals = sorted(vals)
    n = len(vals)
    if n % 2 == 1:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def plot_publication_summary(
    per_model_test_csvs: Dict[str, str],
    per_model_metrics_csvs: Dict[str, str],
    output_path: str,
) -> Optional[str]:
    rows: List[Tuple[str, float, Optional[float], Optional[float], Optional[float]]] = []
    for model, test_csv in per_model_test_csvs.items():
        if not os.path.exists(test_csv):
            continue
        recs = _read_csv(test_csv)
        if not recs:
            continue
        r = recs[0]
        test_loss = _to_float(r.get("test_loss"))
        if test_loss is None:
            continue
        pbal = _to_float(r.get("test_power_balance"))
        vlim = _to_float(r.get("test_voltage_limit"))
        med_t = _median_epoch_time(per_model_metrics_csvs.get(model, ""))
        rows.append((model, test_loss, pbal, vlim, med_t))

    if not rows:
        return None

    rows.sort(key=lambda x: x[1])
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    models = [r[0] for r in rows]
    x = list(range(len(models)))

    # Panel 1: predictive performance
    axes[0].bar(x, [r[1] for r in rows])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=30, ha="right")
    axes[0].set_ylabel("test_loss")
    axes[0].set_title("Predictive Error")

    # Panel 2: physics balance (if available)
    p_available = any(r[2] is not None for r in rows)
    v_available = any(r[3] is not None for r in rows)
    if p_available or v_available:
        width = 0.4
        if p_available:
            axes[1].bar([i - width / 2 for i in x], [r[2] if r[2] is not None else 0.0 for r in rows], width=width, label="test_power_balance")
        if v_available:
            axes[1].bar([i + width / 2 for i in x], [r[3] if r[3] is not None else 0.0 for r in rows], width=width, label="test_voltage_limit")
        axes[1].legend(loc="best", frameon=True)
        axes[1].set_ylabel("constraint metric")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=30, ha="right")
        axes[1].set_title("Physics Terms")
    else:
        axes[1].text(0.5, 0.5, "No physics metrics available", ha="center", va="center")
        axes[1].set_axis_off()

    # Panel 3: training efficiency
    eff_vals = [r[4] if r[4] is not None else 0.0 for r in rows]
    axes[2].bar(x, eff_vals)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=30, ha="right")
    axes[2].set_ylabel("median epoch time (s)")
    axes[2].set_title("Training Efficiency")

    fig.suptitle("Training Summary (Performance, Physics, Efficiency)")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def write_summary_index(index_path: str, payload: Dict[str, object]) -> str:
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return index_path

