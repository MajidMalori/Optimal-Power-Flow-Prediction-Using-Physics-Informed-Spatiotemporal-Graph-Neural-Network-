from typing import List, Dict

from src.visualization.plot_ws_common import apply_ws_style, METHOD_COLORS, save_pub_figure


def plot_ws_feasibility(records: List[Dict], summary: Dict[str, float], case_name: str, output_dir: str) -> None:
    import matplotlib.pyplot as plt

    if not records:
        return

    apply_ws_style()
    methods = ["flat", "dc", "warmstart"]

    # 1) Feasibility rate
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [100.0 * summary.get(f"{m}_feasibility_rate", 0.0) for m in methods]
    ax.bar(methods, vals, color=[METHOD_COLORS[m] for m in methods])
    ax.set_ylim(0, 105)
    ax.set_title(f"{case_name.upper()} - Feasibility Rate")
    ax.set_xlabel("Initialization Method")
    ax.set_ylabel("Feasible Solutions (%)")
    save_pub_figure(fig, output_dir, f"{case_name}_feasibility_rate")
    plt.close(fig)

    # 2) Constraint satisfaction
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [100.0 * summary.get(f"{m}_constraint_rate_mean", 0.0) for m in methods]
    ax.bar(methods, vals, color=[METHOD_COLORS[m] for m in methods])
    ax.set_ylim(0, 105)
    ax.set_title(f"{case_name.upper()} - Mean Constraint Satisfaction")
    ax.set_xlabel("Initialization Method")
    ax.set_ylabel("Constraint Satisfaction (%)")
    save_pub_figure(fig, output_dir, f"{case_name}_constraint_satisfaction")
    plt.close(fig)
