import os
from typing import Dict


METHOD_COLORS: Dict[str, str] = {
    "flat": "#ADB5BD",
    "dc": "#6C757D",
    "warmstart": "#E63946",
}


def ensure_case_dir(base_dir: str, case_name: str) -> str:
    out = os.path.join(base_dir, case_name)
    os.makedirs(out, exist_ok=True)
    return out


def apply_ws_style() -> None:
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 300


def save_pub_figure(fig, output_dir: str, stem: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    png = os.path.join(output_dir, f"{stem}.png")
    fig.savefig(png, dpi=600, bbox_inches="tight")
