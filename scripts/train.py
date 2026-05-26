"""
Training script for Power Flow prediction models.

All settings are configurable via configs/training.yaml.
CLI flags (--case, --models, --offline, --online) override YAML defaults.

Usage:
    python scripts/train.py --case 33 --models all              # Uses YAML default mode (offline)
    python scripts/train.py --case 33 --models all --online     # Force live cloud sync
    python scripts/train.py --case 33 --models StandardGCN      # Train a single model
"""

import os
import sys
import copy
import inspect
import argparse
import logging
import json
from datetime import datetime

# ── Suppress noisy warnings (must happen before library imports) ─────────────
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*Checkpoint directory.*exists and is not empty.*")
warnings.filterwarnings("ignore", message=".*The anonymous setting has no effect.*")
warnings.filterwarnings("ignore", message=".*LeafSpec.*")

os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"
os.environ["LIGHTNING_PYTORCH_DISABLE_TIP"] = "1"

import yaml
import wandb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger

logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

from src.models import SPATIAL_MODELS, RECURRENT_MODELS, PowerFlowDataModule, get_model_registry
from src.training.metrics_recorder import RecorderPaths, TrainingMetricsRecorder
from src.visualization.plot_training import (
    build_case_training_metrics_csv,
    plot_case_final_metrics,
    plot_case_loss_overlay,
    plot_loss_curves,
    plot_publication_summary,
    plot_lr_curve,
    plot_test_metrics,
    plot_timing,
    write_summary_index,
)

MODEL_REGISTRY = get_model_registry()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "prep")


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(config):
    model_cfg = config['model']
    model_name = model_cfg['name']

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Options: {list(MODEL_REGISTRY.keys())}")

    ModelClass = MODEL_REGISTRY[model_name]

    kwargs = {
        'in_channels': model_cfg['in_channels'],
        'out_channels': model_cfg['out_channels'],
        'learning_rate': model_cfg.get('learning_rate', 1e-3),
        'lr_patience': model_cfg.get('lr_patience', 10),
        'lr_factor': model_cfg.get('lr_factor', 0.5)
    }

    if 'gcn_hidden' in model_cfg: kwargs['gcn_hidden'] = model_cfg['gcn_hidden']
    if 'lstm_hidden' in model_cfg: kwargs['lstm_hidden'] = model_cfg['lstm_hidden']
    if 'gru_hidden' in model_cfg: kwargs['gru_hidden'] = model_cfg['gru_hidden']
    if 'hidden_channels' in model_cfg: kwargs['hidden_channels'] = model_cfg['hidden_channels']

    if model_name in SPATIAL_MODELS:
        if 'hidden_channels' not in kwargs and 'gcn_hidden' in kwargs:
            kwargs['hidden_channels'] = kwargs['gcn_hidden']

    if 'num_layers' in model_cfg:
        kwargs['num_layers'] = model_cfg['num_layers']
        kwargs['num_gcn_layers'] = model_cfg['num_layers']
        kwargs['num_res_blocks'] = model_cfg['num_layers']

    # Filter kwargs to match the model's actual __init__ signature
    sig = inspect.signature(ModelClass.__init__)
    valid_params = [p for p in sig.parameters.keys() if p != 'self']
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if not has_kwargs:
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return ModelClass(**filtered_kwargs)

    return ModelClass(**kwargs)


# ── Training ─────────────────────────────────────────────────────────────────
def train_single_model(model_name, case_name, config, group_name):
    """Train and test a single model on a single case. Each call gets its own W&B run."""

    run_config = copy.deepcopy(config)
    run_config['data']['case_name'] = case_name
    run_config['model']['name'] = model_name

    trainer_cfg = run_config.get('trainer', {})
    logger_cfg = run_config.get('logger', {})

    # Spatial models only support seq_len = 1
    actual_seq_len = run_config['data']['seq_len'] if model_name in RECURRENT_MODELS else 1

    # 1. Data
    L.seed_everything(trainer_cfg.get('seed', 42), verbose=False)
    dm = PowerFlowDataModule(
        data_dir=PROCESSED_DIR,
        case_name=case_name,
        batch_size=run_config['data']['batch_size'],
        seq_len=actual_seq_len
    )

    # 2. Model
    model = build_model(run_config)

    # 3. Outputs
    paths = RecorderPaths.for_run(PROJECT_ROOT, case_name, model_name)
    paths.ensure()

    # 4. Loggers
    wandb_logger = WandbLogger(
        project=logger_cfg.get('project', 'powerflow-pinn'),
        name=f"{model_name}_{case_name}",
        group=group_name,
        save_dir=logger_cfg.get('save_dir', 'wandb_logs'),
        config=run_config,
        tags=[model_name, case_name, group_name, "spatial" if model_name in SPATIAL_MODELS else "recurrent"],
        anonymous=False
    )

    csv_logger = CSVLogger(save_dir=paths.logs_dir, name="lightning")

    run_meta = {
        "case": case_name,
        "model": model_name,
        "group": group_name,
        "wandb_mode": os.environ.get("WANDB_MODE", ""),
        "max_epochs": trainer_cfg.get("max_epochs", 100),
        "seq_len": actual_seq_len,
        "batch_size": run_config["data"]["batch_size"],
    }
    recorder_cb = TrainingMetricsRecorder(paths=paths, run_meta=run_meta)

    # 5. Callbacks (checkpoints + early stop + recorder)
    # Each session gets its own folder to prevent accidental overwriting.
    ckpt_dir = os.path.join("checkpoints", group_name, f"{model_name}_{case_name}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=run_config['callbacks']['model_checkpoint']['monitor'],
        filename=run_config['callbacks']['model_checkpoint'].get('filename', '{epoch:02d}-{val_loss:.4f}'),
        save_top_k=run_config['callbacks']['model_checkpoint']['save_top_k'],
        mode=run_config['callbacks']['model_checkpoint']['mode']
    )
    early_stop_cb = EarlyStopping(**run_config['callbacks']['early_stopping'])

    # 6. Trainer (all settings from YAML)
    trainer = L.Trainer(
        logger=[wandb_logger, csv_logger],
        callbacks=[checkpoint_cb, early_stop_cb, recorder_cb],
        max_epochs=trainer_cfg.get('max_epochs', 100),
        accelerator=trainer_cfg.get('accelerator', 'auto'),
        devices=trainer_cfg.get('devices', 'auto'),
        log_every_n_steps=trainer_cfg.get('log_every_n_steps', 10),
        enable_progress_bar=trainer_cfg.get('enable_progress_bar', True),
        enable_model_summary=trainer_cfg.get('enable_model_summary', False),
        num_sanity_val_steps=0
    )

    # 7. Train
    trainer.fit(model, datamodule=dm)

    # 8. Test (scalar metrics only)
    test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
    
    if test_results:
        import pandas as pd
        # Convert list of dicts to DataFrame
        df_test = pd.DataFrame(test_results)
        # Add identifying columns
        df_test.insert(0, "case", case_name)
        df_test.insert(0, "model", model_name)

        per_model_test_csv = os.path.join(paths.csv_dir, "test_metrics.csv")
        df_test.to_csv(per_model_test_csv, index=False)

    # 9. Plots (training-only)
    metrics_csv = recorder_cb.metrics_csv_path
    plot_loss_curves(metrics_csv, os.path.join(paths.figures_dir, "loss_curves.png"))
    plot_lr_curve(metrics_csv, os.path.join(paths.figures_dir, "lr.png"))
    plot_timing(metrics_csv, os.path.join(paths.figures_dir, "timing.png"))
    plot_test_metrics(os.path.join(paths.csv_dir, "test_metrics.csv"), os.path.join(paths.figures_dir, "test_metrics.png"))

    # 10. Close W&B run cleanly before starting the next model
    wandb.finish()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Power Flow Models")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--case", type=str, default=None, help="Comma separated cases (e.g., '33,57' or 'all')")
    parser.add_argument("--models", type=str, default=None, help="Comma separated models or 'all'")
    parser.add_argument("--epochs", type=int, default=None, help="Override training max epochs")
    parser.add_argument("--offline", action="store_true", help="Force W&B offline mode (fast, sync later)")
    parser.add_argument("--online", action="store_true", help="Force W&B online mode (live cloud sync)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.epochs is not None:
        if 'trainer' not in config:
            config['trainer'] = {}
        config['trainer']['max_epochs'] = args.epochs

    # ── Resolve W&B mode: CLI flags override YAML ────────────────────────
    if args.online:
        wandb_mode = "online"
    elif args.offline:
        wandb_mode = "offline"
    else:
        wandb_mode = config.get('logger', {}).get('mode', 'offline')

    os.environ["WANDB_MODE"] = wandb_mode

    # ── Resolve cases ────────────────────────────────────────────────────
    if args.case:
        if args.case.lower() == 'all':
            cases = ['case33', 'case57', 'case118']
        else:
            cases = [c.strip() if c.strip().startswith('case') else f"case{c.strip()}" for c in args.case.split(',')]
    else:
        cases = [config['data']['case_name']]

    # ── Resolve models ───────────────────────────────────────────────────
    if args.models:
        if args.models.lower() == 'all':
            models_to_train = list(MODEL_REGISTRY.keys())
        else:
            models_to_train = [m.strip() for m in args.models.split(',')]
    else:
        models_to_train = [config['model']['name']]

    # ── Training loop ────────────────────────────────────────────────────
    total = len(cases) * len(models_to_train)
    current = 0
    group_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\nTraining {total} model(s) | Mode: {wandb_mode} | Cases: {', '.join(cases)}")

    import shutil
    for case_name in cases:
        # Clear out old training reports for this case
        report_dir = os.path.join(PROJECT_ROOT, "reports", "training", case_name)
        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)
            
        for model_name in models_to_train:
            current += 1
            print(f"\n{'='*60}")
            print(f"  [{current}/{total}] {model_name} on {case_name}")
            print(f"{'='*60}\n")

            try:
                train_single_model(model_name, case_name, config, group_name)
            except Exception as e:
                print(f"\nERROR: {model_name} on {case_name} failed: {e}")
                wandb.finish(exit_code=1)
                continue

        # Case summary (training-only)
        case_dir = os.path.join(PROJECT_ROOT, "reports", "training", case_name)
        summary_dir = os.path.join(case_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        per_model_metrics = {}
        per_model_tests = {}
        for model_name in models_to_train:
            run_paths = RecorderPaths.for_run(PROJECT_ROOT, case_name, model_name)
            per_model_metrics[model_name] = os.path.join(run_paths.csv_dir, "metrics_epoch.csv")
            per_model_tests[model_name] = os.path.join(run_paths.csv_dir, "test_metrics.csv")

        case_csv_dir = os.path.join(case_dir, "csv")
        os.makedirs(case_csv_dir, exist_ok=True)
        master_csv = os.path.join(case_csv_dir, f"{case_name}_training_metrics.csv")
        build_case_training_metrics_csv(per_model_tests, master_csv)

        loss_overlay = plot_case_loss_overlay(per_model_metrics, os.path.join(summary_dir, "loss_overlay.png"))
        test_loss_comp = plot_case_final_metrics(per_model_tests, "test_loss", os.path.join(summary_dir, "test_loss_comparison.png"))
        publication_summary = plot_publication_summary(
            per_model_tests,
            per_model_metrics,
            os.path.join(summary_dir, "publication_summary.png"),
        )

        write_summary_index(
            os.path.join(summary_dir, "index.json"),
            {
                "case": case_name,
                "models": models_to_train,
                "generated_at": datetime.now().isoformat(),
                "files": {
                    "case_metrics_csv": master_csv,
                    "loss_overlay": loss_overlay,
                    "test_loss_comparison": test_loss_comp,
                    "publication_summary": publication_summary,
                },
            },
        )

    print(f"\n{'='*60}")
    print(f"  All done! ({current}/{total} completed)")
    if wandb_mode == "offline":
        print(f"  Sync to cloud: wandb sync {config.get('logger', {}).get('save_dir', 'wandb_logs')}/wandb/latest-run")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
