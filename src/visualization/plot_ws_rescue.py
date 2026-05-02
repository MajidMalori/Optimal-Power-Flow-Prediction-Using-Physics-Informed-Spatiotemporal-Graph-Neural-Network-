from typing import List, Dict

from src.visualization.plot_ws_common import apply_ws_style, METHOD_COLORS, save_pub_figure


def plot_ws_rescue(records: List[Dict], summary: Dict[str, float], case_name: str, output_dir: str) -> None:
    import matplotlib.pyplot as plt

    apply_ws_style()

    # 1) Recovery comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["dc", "warmstart"]
    vals = [
        100.0 * summary.get("dc_recovery_rate", 0.0),
        100.0 * summary.get("warmstart_recovery_rate", 0.0),
    ]
    colors = [METHOD_COLORS["dc"], METHOD_COLORS["warmstart"]]
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 105)
    ax.set_title(f"{case_name.upper()} - Rescue Recovery Rate")
    ax.set_xlabel("Recovery Method")
    ax.set_ylabel("Recovered Candidates (%)")
    save_pub_figure(fig, output_dir, f"{case_name}_rescue_recovery_rate")
    plt.close(fig)

    # 2) Candidate count
    fig, ax = plt.subplots(figsize=(8, 5))
    n = int(summary.get("rescue_candidates", 0.0))
    ax.bar(["flat-failed candidates"], [n], color="#495057")
    ax.set_title(f"{case_name.upper()} - Rescue Candidate Count")
    ax.set_ylabel("Number of Samples")
    save_pub_figure(fig, output_dir, f"{case_name}_rescue_candidates")
    plt.close(fig)
