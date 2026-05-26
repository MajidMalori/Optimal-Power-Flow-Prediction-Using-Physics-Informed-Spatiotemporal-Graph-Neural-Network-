import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import lightning as L


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class RecorderPaths:
    run_dir: str
    csv_dir: str
    figures_dir: str
    logs_dir: str

    @staticmethod
    def for_run(project_root: str, case_name: str, model_name: str) -> "RecorderPaths":
        run_dir = os.path.join(project_root, "reports", "training", case_name, model_name)
        return RecorderPaths(
            run_dir=run_dir,
            csv_dir=os.path.join(run_dir, "csv"),
            figures_dir=os.path.join(run_dir, "figures"),
            logs_dir=os.path.join(run_dir, "logs"),
        )

    def ensure(self) -> None:
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)


class TrainingMetricsRecorder(L.Callback):
    def __init__(self, paths: RecorderPaths, run_meta: Dict[str, Any]):
        super().__init__()
        self.paths = paths
        self.run_meta = run_meta
        self._epoch_t0: Optional[float] = None
        self._rows: list[dict[str, Any]] = []

    @property
    def metrics_csv_path(self) -> str:
        return os.path.join(self.paths.csv_dir, "metrics_epoch.csv")

    @property
    def meta_json_path(self) -> str:
        return os.path.join(self.paths.run_dir, "run_meta.json")

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.paths.ensure()
        with open(self.meta_json_path, "w", encoding="utf-8") as f:
            json.dump(self.run_meta, f, indent=2)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        t1 = time.perf_counter()
        epoch_time_s = (t1 - self._epoch_t0) if self._epoch_t0 is not None else None

        metrics = dict(trainer.callback_metrics)
        row: Dict[str, Any] = {
            "epoch": int(getattr(trainer, "current_epoch", 0)),
            "epoch_time_s": float(epoch_time_s) if epoch_time_s is not None else None,
        }

        for k, v in metrics.items():
            fv = _to_float(v)
            if fv is not None:
                row[str(k)] = fv

        lr = None
        try:
            if trainer.optimizers:
                lr = trainer.optimizers[0].param_groups[0].get("lr", None)
        except Exception:
            lr = None
        if lr is not None:
            row["lr"] = float(lr)

        self._rows.append(row)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._rows:
            return
        keys = sorted({k for r in self._rows for k in r.keys()})
        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._rows)

