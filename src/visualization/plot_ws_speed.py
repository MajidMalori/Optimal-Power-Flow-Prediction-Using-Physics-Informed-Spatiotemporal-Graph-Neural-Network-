from typing import List, Dict

from src.visualization.plot_ws_common import apply_ws_style, METHOD_COLORS, save_pub_figure


def plot_ws_speed(records: List[Dict], summary: Dict[str, float], case_name: str, output_dir: str) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    if not records:
        return

    apply_ws_style()
    df = pd.DataFrame(records)
    df_ok = df[df["converged"] == True]

    # 1) Time distribution
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=df_ok, x="method", y="time_ms", order=["flat", "dc", "warmstart"], palette=METHOD_COLORS, ax=ax)
    ax.set_yscale("log")
    ax.set_title(f"{case_name.upper()} - Solve Time Distribution (log scale)")
    ax.set_xlabel("Initialization Method")
    ax.set_ylabel("Time (ms)")
    save_pub_figure(fig, output_dir, f"{case_name}_speed_time_boxplot")
    plt.close(fig)

    # 2) Iteration means
    fig, ax = plt.subplots(figsize=(10, 5))
    means = [
        summary.get("flat_iter_mean", 0.0),
        summary.get("dc_iter_mean", 0.0),
        summary.get("warmstart_iter_mean", 0.0),
    ]
    methods = ["flat", "dc", "warmstart"]
    ax.bar(methods, means, color=[METHOD_COLORS[m] for m in methods])
    ax.set_title(f"{case_name.upper()} - Mean Newton-Raphson Iterations")
    ax.set_xlabel("Initialization Method")
    ax.set_ylabel("Mean Iterations")
    save_pub_figure(fig, output_dir, f"{case_name}_speed_iterations_bar")
    plt.close(fig)

    # 3) Success rate
    fig, ax = plt.subplots(figsize=(10, 5))
    rates = [
        100.0 * summary.get("flat_success_rate", 0.0),
        100.0 * summary.get("dc_success_rate", 0.0),
        100.0 * summary.get("warmstart_success_rate", 0.0),
    ]
    methods = ["flat", "dc", "warmstart"]
    ax.bar(methods, rates, color=[METHOD_COLORS[m] for m in methods])
    ax.set_ylim(0, 105)
    ax.set_title(f"{case_name.upper()} - Convergence Success Rate")
    ax.set_xlabel("Initialization Method")
    ax.set_ylabel("Success Rate (%)")
    save_pub_figure(fig, output_dir, f"{case_name}_speed_success_rate")
    plt.close(fig)
