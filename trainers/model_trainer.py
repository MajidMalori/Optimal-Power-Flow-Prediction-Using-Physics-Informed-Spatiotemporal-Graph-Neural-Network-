import torch
import os
import logging
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.forensic_logger import get_logger

class PowerSystemTrainer:
    """
    Trainer for Power System Denoising State Estimator.
    Consolidates training logic, logging, and evaluation.
    """
    def __init__(self, model, criterion, optimizer, config, device, is_physics_informed=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.is_physics_informed = is_physics_informed
        self.forensic_logger = get_logger()
        
        # Initialize attributes for tracking training progress
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_epoch = 0
        
        # Logging
        self.log_file = None
        self.log_file_path = None
        
        # History
        self.history = {
            'train_total_loss': [], 'train_mse': [], 'train_mae': [],
            'train_physics_loss': [], 'train_safety_loss': [],
            'val_total_loss': [], 'val_mse': [], 'val_mae': [],
            'val_physics_loss': [], 'val_safety_loss': [],
            'train_weights': [], 'train_log_vars': [],
            'learning_rate': []
        }
        
        # Scheduler initialization (lazy, done in train())
        self.scheduler = None

    def _get_progress_desc(self, metrics, batch_count, weights=None):
        """Helper to generate progress bar description string."""
        avg_mse = metrics['mse'] / batch_count
        avg_total_loss = metrics['total_loss'] / batch_count
        
        show_detailed = getattr(self.config, 'SHOW_DETAILED_PROGRESS', False)
        
        if self.is_physics_informed:
            avg_phys = metrics['physics_loss'] / batch_count
            avg_safe = metrics['safety_loss'] / batch_count
            
            if show_detailed and weights is not None:
                desc = f"L={avg_total_loss:.2f} M={avg_mse:.3f} P={avg_phys:.3f} S={avg_safe:.3f} w=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}]"
            else:
                desc = f"M={avg_mse:.6f} P={avg_phys:.6f} S={avg_safe:.6f}"
        else:
            desc = f"MSE: {avg_mse:.6f}"
            
        return desc

    def _train_epoch(self, train_loader):
        self.model.train()
        
        grad_accum_steps = getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        
        epoch_metrics = {
            'total_loss': 0.0, 'mse': 0.0, 'mae': 0.0,
            'physics_loss': 0.0, 'safety_loss': 0.0,
            'grad_norm': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Train]")
        
        batch_count = 0
        self.optimizer.zero_grad()
        
        self._last_weights = None
        
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            ybus = batch['ybus_matrix'].to(self.device, non_blocking=True)
            adj = batch['adjacency'].to(self.device, non_blocking=True)
            
            outputs = self.model(features, adj)
            loss_dict = self.criterion(
                outputs_norm=outputs,
                targets_norm=targets,
                measurements_norm=features,
                ybus_batch=ybus,
                return_components=True,
                epoch=self.current_epoch
            )
            loss = loss_dict['total_loss']
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            
            # Metrics update (scale back)
            scaled_loss = loss_dict['total_loss'].item()
            epoch_metrics['total_loss'] += scaled_loss
            epoch_metrics.update({k: epoch_metrics[k] + loss_dict[k] for k in epoch_metrics if k in loss_dict and k != 'grad_norm'})
            
            if self.is_physics_informed:
                self._last_weights = loss_dict.get('weights')
            
            if not self.is_physics_informed:
                mae_batch = torch.nn.functional.l1_loss(outputs, targets).item()
                epoch_metrics['mae'] += mae_batch
            
            batch_count += 1
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                epoch_metrics['grad_norm'] += grad_norm.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Cleanup
            del features, targets, ybus, adj, outputs, loss_dict, loss
            
            # Update pbar
            if batch_count % max(1, len(train_loader) // 10) == 0 or batch_idx == len(train_loader) - 1:
                desc = self._get_progress_desc(epoch_metrics, batch_count, weights=self._last_weights)
                pbar.set_postfix_str(desc)
                if self.is_physics_informed and not getattr(self.config, 'SHOW_DETAILED_PROGRESS', False):
                    self._last_weights = None # Optimize memory
        
        num_batches = len(train_loader)
        result = {k: v / num_batches for k, v in epoch_metrics.items()}
        result['weights'] = self._last_weights if self.is_physics_informed else None
        
        return result

    def _val_epoch(self, val_loader):
        self.model.eval()
        epoch_metrics = {
            'total_loss': 0.0, 'mse': 0.0, 'mae': 0.0,
            'physics_loss': 0.0, 'safety_loss': 0.0
        }
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Val]")
        
        batch_count = 0
        last_weights = None
        
        with torch.no_grad():
            for batch in pbar:
                features = batch['features'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)
                ybus = batch['ybus_matrix'].to(self.device, non_blocking=True)
                adj = batch['adjacency'].to(self.device, non_blocking=True)
                
                outputs = self.model(features, adj)
                loss_dict = self.criterion(
                    outputs_norm=outputs,
                    targets_norm=targets,
                    measurements_norm=features,
                    ybus_batch=ybus,
                    return_components=True
                )
                
                epoch_metrics.update({k: epoch_metrics[k] + loss_dict[k] for k in epoch_metrics if k in loss_dict})
                
                if self.is_physics_informed:
                    last_weights = loss_dict.get('weights')
                
                if not self.is_physics_informed:
                    mae_batch = torch.nn.functional.l1_loss(outputs, targets).item()
                    epoch_metrics['mae'] += mae_batch
                
                batch_count += 1
                
                desc = self._get_progress_desc(epoch_metrics, batch_count, weights=last_weights)
                pbar.set_postfix_str(desc)
                del features, targets, ybus, adj, outputs, loss_dict
        
        num_batches = len(val_loader)
        result = {k: v / num_batches for k, v in epoch_metrics.items()}
        result['weights'] = last_weights if self.is_physics_informed else None
        
        return result

    def train(self, train_loader, val_loader, model_name=None, num_buses=None, config_params=None):
        """Main training loop."""
        # Initialize logging
        debug_enabled = getattr(self.config, 'DEBUG_ENABLE', False)
        if debug_enabled and hasattr(self.config, 'get_training_log_path') and model_name and num_buses:
            mode = getattr(self.config, 'DATA_MODE', 'train')
            self.log_file_path = self.config.get_training_log_path(num_buses, model_name, mode)
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            
            # Check existence for header
            file_exists = os.path.exists(self.log_file_path)
            self.log_file = open(self.log_file_path, 'a', encoding='utf-8')
            
            if not file_exists:
                self.log_file.write(f"{'='*80}\nTraining Log for {model_name} ({num_buses}-bus) - {mode.upper()} Mode\n{'='*80}\n{datetime.now()}\n{'='*80}\n\n")
            
            self.log_file.write(f"\n{'#'*80}\n# Run Started: {datetime.now()}\n{'#'*80}\n")
            if config_params:
                self.log_file.write(f"\nConfiguration:\n{'-'*80}\n")
                for k, v in sorted(config_params.items()):
                    self.log_file.write(f"  {k:30s}: {v}\n")
                self.log_file.write(f"{'-'*80}\n\n")
        
        def _log(msg):
            if self.log_file:
                try:
                    self.log_file.write(f"{msg}\n")
                    self.log_file.flush()
                except Exception:
                    pass

        # Scheduler
        use_scheduler = getattr(self.config, 'USE_LEARNING_RATE_SCHEDULER', True)
        if use_scheduler:
            t_max = getattr(self.config, 'COSINEANNEALINGLR_T_MAX', self.config.NUM_EPOCHS)
            eta_min = getattr(self.config, 'COSINEANNEALINGLR_ETA_MIN', 1e-5)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
            _log(f"[Config] Scheduler: CosineLR(T_max={t_max}, eta_min={eta_min:.2e})")
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        _log(f"[Config] Trainable Params: {trainable_params:,}")
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            # Check shutdown
            try:
                from utils.shutdown_flag import get_shutdown
                if get_shutdown():
                    _log("Shutdown signal received.")
                    break
            except ImportError:
                pass

            self.current_epoch = epoch
            train_metrics = self._train_epoch(train_loader)
            
            # History update
            self.history['train_total_loss'].append(train_metrics['total_loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['train_mae'].append(train_metrics.get('mae', 0.0))
            self.history['train_physics_loss'].append(train_metrics['physics_loss'])
            self.history['train_safety_loss'].append(train_metrics['safety_loss'])
            self.history['train_weights'].append(train_metrics.get('weights'))
            
            if hasattr(self.criterion, 'log_vars'):
                self.history['train_log_vars'].append([v.item() for v in self.criterion.log_vars])
            else:
                self.history['train_log_vars'].append(None)
                
            val_metrics = self._val_epoch(val_loader)
            
            self.history['val_total_loss'].append(val_metrics['total_loss'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['val_mae'].append(val_metrics.get('mae', 0.0))
            self.history['val_physics_loss'].append(val_metrics['physics_loss'])
            self.history['val_safety_loss'].append(val_metrics['safety_loss'])
            
            # Forensic Logging
            if self.forensic_logger and self.forensic_logger.enabled:
                log_weights_interval = getattr(self.config, 'DEBUG_LOG_WEIGHTS_EVERY_N_EPOCHS', 10)
                if epoch % log_weights_interval == 0:
                    self.forensic_logger.log_model_weights(self.model, epoch)
                self.forensic_logger.log_epoch_summary(epoch, train_metrics, val_metrics)
            
            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Logging
            val_loss = val_metrics['total_loss']
            train_mse = train_metrics['mse']
            val_mse = val_metrics['mse']
            gap = val_mse - train_mse
            
            weights = train_metrics.get('weights')
            w_str = f"w=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}]" if weights else "w=N/A"
            
            if self.is_physics_informed:
                 _log(f"Epoch {epoch} | Train: MSE={train_mse:.6f}, Phys={train_metrics['physics_loss']:.6f} | Val: MSE={val_mse:.6f}, Phys={val_metrics['physics_loss']:.6f} | {w_str} | LR={current_lr:.6f} | Gap={gap:+.6f}")
            else:
                 _log(f"Epoch {epoch} | Train: MSE={train_mse:.6f} | Val: MSE={val_mse:.6f} | LR={current_lr:.6f} | Gap={gap:+.6f}")

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.best_epoch = epoch
                self._save_checkpoint('best_model.pth')
                _log(f"  ✓ Best model updated. Val Loss: {self.best_val_loss:.6f}")
            else:
                self.epochs_no_improve += 1
                
            if self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                _log(f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                break
        
        # Close log
        if self.log_file:
            self.log_file.write(f"\nTraining completed: {datetime.now()}\n{'='*80}\n")
            self.log_file.close()
            self.log_file = None

    def _save_checkpoint(self, filename):
        if not getattr(self.config, 'SAVE_CHECKPOINTS', True):
            return
        if hasattr(self.config, 'get_checkpoint_path'):
            path = self.config.get_checkpoint_path(filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)

    def get_training_history(self):
        return self.history
