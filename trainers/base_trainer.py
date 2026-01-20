import torch
from abc import ABC, abstractmethod
import os
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR

class BaseTrainer(ABC):
    """
    Abstract base class for trainers. This class defines the main training loop,
    including epoch iteration and early stopping logic, which is inherited by
    all specific trainer implementations.
    """
    def __init__(self, model, criterion, optimizer, config, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # Scheduler will be initialized in train() method when we have access to train_loader
        # Store train loader for scheduler initialization
        self.scheduler = None
        self.scheduler_type = None
        
        # Initialize attributes for tracking training progress
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_epoch = 0  # Track best epoch for improved early stopping
        
        # Initialize training log file (if config provides path)
        self.log_file = None
        if hasattr(config, 'get_training_log_path'):
            # Will be initialized in train() method when we have model_name and num_buses
            self.log_file_path = None
        else:
            self.log_file_path = None

        # Add history tracking
        self.history = {
            'train_total_loss': [], 'train_mse': [], 'train_mae': [],
            'train_physics_loss': [], 'train_safety_loss': [],
            'val_total_loss': [], 'val_mse': [], 'val_mae': [],
            'val_physics_loss': [], 'val_safety_loss': [],
            'train_weights': [], 'train_log_vars': [],
            'learning_rate': []  # Track learning rate for both physics and non-physics models
        }

    @abstractmethod
    def _train_epoch(self, train_loader):
        """Logic for a single training epoch. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _val_epoch(self, val_loader):
        """Logic for a single validation epoch. Must be implemented by subclasses."""
        raise NotImplementedError

    def train(self, train_loader, val_loader, model_name=None, num_buses=None, config_params=None):
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name of the model
            num_buses: Number of buses in the system
            config_params: Dictionary of configuration parameters for this run (for logging)
        """
        # Initialize training log file if config supports it
        debug_enabled = getattr(self.config, 'DEBUG_ENABLE', False)
        if debug_enabled and hasattr(self.config, 'get_training_log_path') and model_name is not None and num_buses is not None:
            # Get mode from config (train/test)
            mode = getattr(self.config, 'DATA_MODE', 'train')
            self.log_file_path = self.config.get_training_log_path(num_buses, model_name, mode)
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            
            # Check if file exists to determine if we should write header or append separator
            file_exists = os.path.exists(self.log_file_path)
            
            # Use UTF-8 encoding to handle Unicode characters (e.g., δ, σ, λ)
            # Use append mode ('a') to accumulate all configurations in one file
            self.log_file = open(self.log_file_path, 'a', encoding='utf-8')
            
            if not file_exists:
                # First time writing to this file - write header
                self.log_file.write(f"{'='*80}\n")
                self.log_file.write(f"Training Log for {model_name} ({num_buses}-bus) - {mode.upper()} Mode\n")
                self.log_file.write(f"{'='*80}\n")
                self.log_file.write(f"File created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.write(f"{'='*80}\n\n")
            
            # Write configuration separator for this training run
            self.log_file.write(f"\n{'#'*80}\n")
            self.log_file.write(f"# Configuration Run - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write(f"{'#'*80}\n")
            
            # Write configuration parameters if provided
            if config_params:
                self.log_file.write(f"\nConfiguration Parameters:\n")
                self.log_file.write(f"{'-'*80}\n")
                for key, value in sorted(config_params.items()):
                    # Format parameter nicely
                    if isinstance(value, float):
                        self.log_file.write(f"  {key:30s}: {value:.6f}\n")
                    elif isinstance(value, int):
                        self.log_file.write(f"  {key:30s}: {value}\n")
                    else:
                        self.log_file.write(f"  {key:30s}: {value}\n")
                self.log_file.write(f"{'-'*80}\n\n")
            
            self.log_file.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write(f"{'='*80}\n\n")
        else:
            self.log_file = None
        
        def _log(message):
            """Write to log file if available, otherwise do nothing (silent)."""
            if self.log_file:
                try:
                    self.log_file.write(f"{message}\n")
                    self.log_file.flush()
                except UnicodeEncodeError:
                    # Fallback: replace Unicode characters if encoding fails (shouldn't happen with UTF-8, but safety net)
                    safe_message = message.replace('δ', 'delta').replace('σ', 'sigma').replace('λ', 'lambda')
                    self.log_file.write(f"{safe_message}\n")
                    self.log_file.flush()
        
        # Golden Configuration: Use CosineAnnealingLR only
        use_scheduler = getattr(self.config, 'USE_LEARNING_RATE_SCHEDULER', True)
        
        if use_scheduler:
            t_max = getattr(self.config, 'COSINEANNEALINGLR_T_MAX', None)
            if t_max is None:
                t_max = self.config.NUM_EPOCHS  # One full cosine cycle
            eta_min = getattr(self.config, 'COSINEANNEALINGLR_ETA_MIN', 1e-5)  # Higher min LR (was 1e-6) - prevents premature convergence
            
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min
            )
            self.scheduler_type = 'CosineAnnealingLR'
        else:
            self.scheduler = None
            self.scheduler_type = None
        
        # Print scheduler info
        scheduler_info = ""
        if hasattr(self, 'scheduler') and self.scheduler is not None and hasattr(self, 'scheduler_type'):
            if self.scheduler_type == 'CosineAnnealingLR':
                t_max = getattr(self.config, 'COSINEANNEALINGLR_T_MAX', self.config.NUM_EPOCHS)
                eta_min = getattr(self.config, 'COSINEANNEALINGLR_ETA_MIN', 1e-6)
                scheduler_info = f"CosineLR(T_max={t_max}, eta_min={eta_min:.2e})"
            else:
                scheduler_info = "None"
        else:
            scheduler_info = "No scheduler"
        
        # Log config info to file (not terminal - cleaner training output)
        _log(f"[Config] {scheduler_info}")
        
        # Log detailed training configuration for overfitting analysis
        _log(f"[Training Config] Batch Size: {self.config.BATCH_SIZE} | Learning Rate: {self.config.LEARNING_RATE} | Epochs: {self.config.NUM_EPOCHS}")
        _log(f"[Training Config] Early Stopping Patience: {self.config.EARLY_STOPPING_PATIENCE} | Weight Decay: {getattr(self.config, 'WEIGHT_DECAY', 0.0)}")
        
        # Get model dropout (check common attributes)
        model_dropout = 'N/A'
        if hasattr(self.model, 'dropout'):
            model_dropout = self.model.dropout
        elif hasattr(self.model, 'dropout_rate'):
            model_dropout = self.model.dropout_rate
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'dropout'):
            model_dropout = self.model.module.dropout
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        _log(f"[Training Config] Dropout: {model_dropout} | Trainable Params: {trainable_params:,} | Total Params: {total_params:,}")
        _log(f"[Training Config] Optimizer: {type(self.optimizer).__name__} | Scheduler: {scheduler_info}")
        _log(f"[Training Config] Gradient Clipping: max_norm=1.0 (enabled)")
        _log("="*80)
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            # Check for shutdown flag (set by signal handler)
            try:
                from utils.shutdown_flag import get_shutdown
                if get_shutdown():
                    _log("Shutdown signal received - finishing current epoch and exiting gracefully")
                    break
            except (ImportError, AttributeError):
                pass  # If shutdown module not available, continue training
            
            self.current_epoch = epoch
            
            train_metrics = self._train_epoch(train_loader)

            self.history['train_total_loss'].append(train_metrics['total_loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['train_mae'].append(train_metrics.get('mae', 0.0))
            self.history['train_physics_loss'].append(train_metrics['physics_loss'])
            self.history['train_safety_loss'].append(train_metrics['safety_loss'])
            # Track Kendall's learned weights and log_vars
            if 'weights' in train_metrics:
                self.history['train_weights'].append(train_metrics['weights'])
            else:
                self.history['train_weights'].append(None)
            # Track log_vars if available (for debugging overfitting)
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
            
            # Forensic: Log model weights periodically
            if hasattr(self, 'forensic_logger') and self.forensic_logger and self.forensic_logger.enabled:
                log_weights_interval = getattr(self.config, 'DEBUG_LOG_WEIGHTS_EVERY_N_EPOCHS', 10)
                if epoch % log_weights_interval == 0:
                    self.forensic_logger.log_model_weights(self.model, epoch)

            val_loss = val_metrics['total_loss']
            
            # Step CosineAnnealingLR scheduler (per-epoch)
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Track learning rate in history
            self.history['learning_rate'].append(current_lr)
            
            # Calculate overfitting metrics
            train_mse = train_metrics['mse']
            val_mse = val_metrics['mse']
            overfitting_gap = val_mse - train_mse  # Positive = overfitting
            overfitting_ratio = val_mse / train_mse if train_mse > 0 else float('inf')
            
            # Get gradient norm from training metrics (tracked during training)
            avg_grad_norm = train_metrics.get('grad_norm', 0.0)
            
            # Get weight decay value
            weight_decay = self.optimizer.param_groups[0].get('weight_decay', 0.0)
            
            # Log complete epoch summary to file (one line per epoch)
            if hasattr(self, 'criterion') and self.criterion.is_physics_informed:
                # Physics-informed model: include MSE, physics, safety, weights, log_vars, and LR
                train_phys = train_metrics['physics_loss']
                train_safe = train_metrics['safety_loss']
                val_phys = val_metrics['physics_loss']
                val_safe = val_metrics['safety_loss']
                
                # Extract Kendall's learned weights
                weights = train_metrics.get('weights')
                weight_str = f"w=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}]" if weights is not None else "w=N/A"
                
                # Extract log_vars for debugging (these are added to total loss and can cause overfitting)
                log_vars_str = ""
                if hasattr(self.criterion, 'log_vars'):
                    log_vars = [v.item() for v in self.criterion.log_vars]
                    log_vars_str = f" | log_vars=[{log_vars[0]:.3f},{log_vars[1]:.3f},{log_vars[2]:.3f}]"
                
                _log(f"Epoch {epoch} | Train: MSE={train_mse:.6f}, Phys={train_phys:.6f}, Safe={train_safe:.6f} | Val: MSE={val_mse:.6f}, Phys={val_phys:.6f}, Safe={val_safe:.6f} | {weight_str}{log_vars_str} | LR={current_lr:.6f} | Gap={overfitting_gap:+.6f} | Ratio={overfitting_ratio:.3f} | GradNorm={avg_grad_norm:.4f} | WD={weight_decay:.6f}")
            else:
                # Non-physics model: only MSE and LR
                _log(f"Epoch {epoch} | Train: MSE={train_mse:.6f} | Val: MSE={val_mse:.6f} | LR={current_lr:.6f} | Gap={overfitting_gap:+.6f} | Ratio={overfitting_ratio:.3f} | GradNorm={avg_grad_norm:.4f} | WD={weight_decay:.6f}")
            
            # Forensic: Log epoch summary
            if hasattr(self, 'forensic_logger') and self.forensic_logger and self.forensic_logger.enabled:
                self.forensic_logger.log_epoch_summary(epoch, train_metrics, val_metrics)
            
            # Early stopping logic: Track best model on ANY improvement (no threshold)
            # Deep learning progress is often incremental - small improvements matter
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.best_epoch = epoch  # ALWAYS update the best epoch on ANY improvement
                self._save_checkpoint('best_model.pth')
                _log(f"  ✓ Best model updated (improvement: {improvement:.6f}) | Best Val Loss: {self.best_val_loss:.6f}")
            else:
                self.epochs_no_improve += 1
                _log(f"  ⚠ No improvement ({self.epochs_no_improve}/{self.config.EARLY_STOPPING_PATIENCE}) | Best: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
            
            # Early stopping: Stop if no improvement for patience epochs
            if self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                message = f"Early stopping: No improvement for {self.epochs_no_improve} epochs (best at epoch {self.best_epoch}, val_loss={self.best_val_loss:.6f})."
                _log(message)
                break
        
        # Final training summary
        _log("="*80)
        _log(f"[Training Summary] Completed {self.current_epoch} epochs")
        _log(f"[Training Summary] Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        if self.current_epoch > 1:
            final_train_mse = train_metrics['mse']
            final_val_mse = val_metrics['mse']
            final_gap = final_val_mse - final_train_mse
            final_ratio = final_val_mse / final_train_mse if final_train_mse > 0 else float('inf')
            _log(f"[Training Summary] Final Train MSE: {final_train_mse:.6f} | Final Val MSE: {final_val_mse:.6f}")
            _log(f"[Training Summary] Final Overfitting Gap: {final_gap:+.6f} | Final Ratio: {final_ratio:.3f}")
        _log("="*80)
        
        # Close log file if it was opened
        # Note: Each training run is a separate trainer instance, so closing is safe.
        # The next configuration will open the file again in append mode and continue.
        if self.log_file:
            self.log_file.write(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_file.write(f"{'='*80}\n")
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None

    def _save_checkpoint(self, filename):
        """Helper function to save model checkpoints."""
        # Only save if explicitly enabled in config
        if not getattr(self.config, 'SAVE_CHECKPOINTS', True):
            return
            
        if hasattr(self.config, 'get_checkpoint_path'):
            path = self.config.get_checkpoint_path(filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)

    def get_training_history(self):
        """Return the training history dictionary."""
        return self.history