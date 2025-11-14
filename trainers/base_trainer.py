import torch
from abc import ABC, abstractmethod
import os

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
        # This allows us to calculate total_steps for OneCycleLR
        self.scheduler = None
        self.scheduler_type = None
        
        # Initialize attributes for tracking training progress
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_epoch = 0  # Track best epoch for improved early stopping

        # Add history tracking
        self.history = {
            'train_total_loss': [], 'train_mse': [], 
            'train_mse_var1': [], 'train_mse_var2': [],
            'train_power_violation': [], 'train_voltage_violation': [],
            'val_total_loss': [], 'val_mse': [],
            'val_mse_var1': [], 'val_mse_var2': [],
            'val_power_violation': [], 'val_voltage_violation': [],
            'sigma_data': [], 'sigma_power': [], 'sigma_voltage': [],
            'effective_lambda_p': [], 'effective_lambda_v': []
        }

    @abstractmethod
    def _train_epoch(self, train_loader):
        """Logic for a single training epoch. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _val_epoch(self, val_loader):
        """Logic for a single validation epoch. Must be implemented by subclasses."""
        raise NotImplementedError

    def train(self, train_loader, val_loader):
        """Main training loop."""
        # Initialize learning rate scheduler
        # Priority: LR_SCHEDULER_TYPE > USE_LEARNING_RATE_SCHEDULER (for backward compatibility)
        scheduler_type_config = getattr(self.config, 'LR_SCHEDULER_TYPE', None)
        use_scheduler_legacy = getattr(self.config, 'USE_LEARNING_RATE_SCHEDULER', False)
        
        # Determine scheduler type
        if scheduler_type_config:
            # Use explicit scheduler type from config (applies to all models)
            scheduler_type = scheduler_type_config
        elif use_scheduler_legacy:
            # Legacy: USE_LEARNING_RATE_SCHEDULER=True means OneCycleLR
            scheduler_type = 'OneCycleLR'
        else:
            # No scheduler
            scheduler_type = None
        
        # Initialize scheduler based on type
        if scheduler_type == 'OneCycleLR':
            from torch.optim.lr_scheduler import OneCycleLR
            
            # Calculate total steps for OneCycleLR
            steps_per_epoch = len(train_loader)
            total_steps = steps_per_epoch * self.config.NUM_EPOCHS
            
            # OneCycleLR parameters
            initial_lr = self.config.LEARNING_RATE
            max_lr_config = getattr(self.config, 'ONECYCLE_MAX_LR', None)
            if max_lr_config is None:
                max_lr = initial_lr * 10  # Default: 10x initial LR
            else:
                max_lr = max_lr_config
            pct_start = getattr(self.config, 'ONECYCLE_PCT_START', 0.3)
            div_factor = getattr(self.config, 'ONECYCLE_DIV_FACTOR', 25.0)
            final_div_factor = getattr(self.config, 'ONECYCLE_FINAL_DIV_FACTOR', 10000.0)
            
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                anneal_strategy='cos'
            )
            self.scheduler_type = 'OneCycleLR'
            
        elif scheduler_type == 'StepLR':
            from torch.optim.lr_scheduler import StepLR
            
            step_size = getattr(self.config, 'STEPLR_STEP_SIZE', 7)
            gamma = getattr(self.config, 'STEPLR_GAMMA', 0.5)
            
            self.scheduler = StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
            self.scheduler_type = 'StepLR'
            
        elif scheduler_type == 'CosineAnnealingLR':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            t_max = getattr(self.config, 'COSINEANNEALINGLR_T_MAX', self.config.NUM_EPOCHS)
            eta_min = getattr(self.config, 'COSINEANNEALINGLR_ETA_MIN', 1e-6)
            
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min
            )
            self.scheduler_type = 'CosineAnnealingLR'
            
        elif scheduler_type == 'ExponentialLR':
            from torch.optim.lr_scheduler import ExponentialLR
            
            gamma = getattr(self.config, 'EXPONENTIALLR_GAMMA', 0.95)
            
            self.scheduler = ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
            self.scheduler_type = 'ExponentialLR'
            
        elif scheduler_type == 'ReduceLROnPlateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            
            mode = getattr(self.config, 'REDUCELRONPLATEAU_MODE', 'min')
            factor = getattr(self.config, 'REDUCELRONPLATEAU_FACTOR', 0.5)
            patience = getattr(self.config, 'REDUCELRONPLATEAU_PATIENCE', 5)
            min_lr = getattr(self.config, 'REDUCELRONPLATEAU_MIN_LR', 1e-6)
            
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr
            )
            self.scheduler_type = 'ReduceLROnPlateau'
            
        else:
            self.scheduler = None
            self.scheduler_type = None
        
        # Initialize Empirical Bayes optimizer if enabled and using heteroscedastic mode
        use_eb = getattr(self.config, 'USE_EMPIRICAL_BAYES', False)
        use_heteroscedastic = getattr(self.config, 'USE_HETEROSCEDASTIC_UNCERTAINTY', False)
        
        if use_eb and use_heteroscedastic:
            from utils.empirical_bayes import EmpiricalBayesOptimizer
            self.eb_optimizer = EmpiricalBayesOptimizer(
                model=self.model,
                config=self.config,
                device=self.device,
                burn_in_epochs=getattr(self.config, 'EB_BURN_IN_EPOCHS', 100),
                update_frequency=getattr(self.config, 'EB_UPDATE_FREQUENCY', 50),
                hyperparameter_steps=getattr(self.config, 'EB_HYPERPARAMETER_STEPS', 50),
                hyperparameter_lr=getattr(self.config, 'EB_HYPERPARAMETER_LR', 0.01)
            )
        else:
            self.eb_optimizer = None
        
        # Print scheduler and EB info in one concise line
        scheduler_info = ""
        if hasattr(self, 'scheduler') and self.scheduler is not None and hasattr(self, 'scheduler_type'):
            if self.scheduler_type == 'StepLR':
                step_size = getattr(self.config, 'STEPLR_STEP_SIZE', 7)
                gamma = getattr(self.config, 'STEPLR_GAMMA', 0.5)
                scheduler_info = f"StepLR(step={step_size}, γ={gamma})"
            elif self.scheduler_type == 'CosineAnnealingLR':
                t_max = getattr(self.config, 'COSINEANNEALINGLR_T_MAX', self.config.NUM_EPOCHS)
                eta_min = getattr(self.config, 'COSINEANNEALINGLR_ETA_MIN', 1e-6)
                scheduler_info = f"CosineLR(T_max={t_max}, η_min={eta_min:.2e})"
            elif self.scheduler_type == 'OneCycleLR':
                max_lr = getattr(self.scheduler, 'max_lr', self.config.LEARNING_RATE * 10)
                scheduler_info = f"OneCycleLR(max_lr={max_lr:.6f})"
            elif self.scheduler_type == 'ReduceLROnPlateau':
                factor = getattr(self.config, 'REDUCELRONPLATEAU_FACTOR', 0.5)
                patience = getattr(self.config, 'REDUCELRONPLATEAU_PATIENCE', 5)
                scheduler_info = f"ReduceLR(factor={factor}, patience={patience})"
            elif self.scheduler_type == 'ExponentialLR':
                gamma = getattr(self.config, 'EXPONENTIALLR_GAMMA', 0.95)
                scheduler_info = f"ExpLR(γ={gamma})"
            else:
                scheduler_info = f"{self.scheduler_type}"
        else:
            scheduler_info = "No scheduler"
        
        eb_info = ""
        if use_eb and use_heteroscedastic and self.eb_optimizer is not None:
            eb_info = f" | EB(burn={self.eb_optimizer.burn_in_epochs}, freq={self.eb_optimizer.update_frequency})"
        
        print(f"[Config] {scheduler_info}{eb_info}")
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            self.current_epoch = epoch
            
            train_metrics = self._train_epoch(train_loader)

            self.history['train_total_loss'].append(train_metrics['loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['train_mse_var1'].append(train_metrics.get('mse_var1', 0.0))
            self.history['train_mse_var2'].append(train_metrics.get('mse_var2', 0.0))
            self.history['train_power_violation'].append(train_metrics['power_violation'])
            self.history['train_voltage_violation'].append(train_metrics['voltage_violation'])
            
            # Update Empirical Bayes hyperparameters if enabled
            if self.eb_optimizer is not None:
                self.eb_optimizer.update_hyperparameters(train_loader, self.criterion, epoch)
            
            val_metrics = self._val_epoch(val_loader)

            self.history['val_total_loss'].append(val_metrics['loss'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['val_mse_var1'].append(val_metrics.get('mse_var1', 0.0))
            self.history['val_mse_var2'].append(val_metrics.get('mse_var2', 0.0))
            self.history['val_power_violation'].append(val_metrics['power_violation'])
            self.history['val_voltage_violation'].append(val_metrics['voltage_violation'])
            
            if hasattr(self, 'criterion') and hasattr(self.criterion, 'use_heteroscedastic'):
                if self.criterion.use_heteroscedastic:
                    # Heteroscedastic: Data loss uses natural parametrization (per-sample uncertainty)
                    # Physics losses use Kendall-style learnable weights (global parameters)
                    loss_type_display = 'Natural + Kendall'  # Natural Parametrization (data) + Kendall (physics)
                    
                    # Data loss: per-sample uncertainty (not a single global parameter)
                    sigma_data = 1.0  # Placeholder (heteroscedastic: per-sample, varies with input)
                    
                    # Only log physics loss sigmas if this is a physics-informed model
                    if self.criterion.is_physics_informed:
                        # Physics losses: Kendall-style learnable weights (same as homoscedastic)
                        sigma_power = torch.exp(self.criterion.log_sigma_power).item()
                        sigma_voltage = torch.exp(self.criterion.log_sigma_voltage).item()
                        
                        effective_lambda_p = 1.0 / (2.0 * sigma_power ** 2)
                        effective_lambda_v = 1.0 / (2.0 * sigma_voltage ** 2)
                        
                        self.history['sigma_data'].append(sigma_data)
                        self.history['sigma_power'].append(sigma_power)
                        self.history['sigma_voltage'].append(sigma_voltage)
                        self.history['effective_lambda_p'].append(effective_lambda_p)
                        self.history['effective_lambda_v'].append(effective_lambda_v)
                        
                        if epoch % 5 == 0 or epoch == 1:
                            print(f"  Learnable σ (power, voltage): ({sigma_power:.4f}, {sigma_voltage:.4f}) | Effective λ: ({effective_lambda_p:.4f}, {effective_lambda_v:.4f})")
                    else:
                        # Non-physics model: no physics losses, so no physics sigmas to display
                        self.history['sigma_data'].append(sigma_data)
                        self.history['sigma_power'].append(0.0)  # Not applicable
                        self.history['sigma_voltage'].append(0.0)  # Not applicable
                        self.history['effective_lambda_p'].append(0.0)  # Not applicable
                        self.history['effective_lambda_v'].append(0.0)  # Not applicable
                elif hasattr(self.criterion, 'log_sigma_data') and self.criterion.log_sigma_data is not None:
                    # Homoscedastic: all uncertainties are learnable parameters
                    sigma_data = torch.exp(self.criterion.log_sigma_data).item()
                    
                    # Only log physics loss sigmas if this is a physics-informed model
                    if self.criterion.is_physics_informed:
                        sigma_power = torch.exp(self.criterion.log_sigma_power).item()
                        sigma_voltage = torch.exp(self.criterion.log_sigma_voltage).item()
                        
                        effective_lambda_p = 1.0 / (2.0 * sigma_power ** 2)
                        effective_lambda_v = 1.0 / (2.0 * sigma_voltage ** 2)
                        
                        self.history['sigma_data'].append(sigma_data)
                        self.history['sigma_power'].append(sigma_power)
                        self.history['sigma_voltage'].append(sigma_voltage)
                        self.history['effective_lambda_p'].append(effective_lambda_p)
                        self.history['effective_lambda_v'].append(effective_lambda_v)
                        
                        if epoch % 5 == 0 or epoch == 1:
                            print(f"  Learnable σ (data, power, voltage): ({sigma_data:.4f}, {sigma_power:.4f}, {sigma_voltage:.4f}) | Effective λ: ({effective_lambda_p:.4f}, {effective_lambda_v:.4f})")
                    else:
                        # Non-physics model: only data loss exists
                        self.history['sigma_data'].append(sigma_data)
                        self.history['sigma_power'].append(0.0)  # Not applicable
                        self.history['sigma_voltage'].append(0.0)  # Not applicable
                        self.history['effective_lambda_p'].append(0.0)  # Not applicable
                        self.history['effective_lambda_v'].append(0.0)  # Not applicable
                        
                        if epoch % 5 == 0 or epoch == 1:
                            print(f"  Learnable σ (data): ({sigma_data:.4f})")

            val_loss = val_metrics.get('loss', float('inf'))
            
            # Step learning rate scheduler if enabled
            # OneCycleLR is stepped per-batch (already done in _train_epoch), just print LR here
            # Other schedulers (StepLR, CosineAnnealingLR, etc.) are stepped per-epoch
            if self.scheduler is not None:
                if self.scheduler_type == 'OneCycleLR':
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if epoch % 5 == 0 or epoch == 1:  # Print LR every 5 epochs or first epoch
                        print(f"  Learning rate: {current_lr:.6f}")
                elif self.scheduler_type == 'ReduceLROnPlateau':
                    # ReduceLROnPlateau steps based on validation loss
                    self.scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if epoch % 5 == 0 or epoch == 1:
                        print(f"  Learning rate: {current_lr:.6f} (val_loss: {val_loss:.6f})")
                else:
                    # StepLR, CosineAnnealingLR, ExponentialLR: step per-epoch
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if epoch % 5 == 0 or epoch == 1:
                        print(f"  Learning rate: {current_lr:.6f}")
            
            if epoch > 1:
                relative_improvement = (self.best_val_loss - val_loss) / (self.best_val_loss + 1e-10)
            else:
                relative_improvement = 0.01
            
            if val_loss < self.best_val_loss:
                if relative_improvement > 0.01:
                    self.best_epoch = epoch
                    
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self._save_checkpoint('best_model.pth')
            else:
                self.epochs_no_improve += 1
                
                epochs_since_significant = epoch - self.best_epoch
                if epochs_since_significant >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping: No >1% improvement for {epochs_since_significant} epochs (best at epoch {self.best_epoch}).")
                    break
                elif self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch} epochs.")
                    break

    def _save_checkpoint(self, filename):
        """Helper function to save model checkpoints."""
        if hasattr(self.config, 'get_checkpoint_path'):
            path = self.config.get_checkpoint_path(filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)

    def get_training_history(self):
        """Return the training history dictionary."""
        return self.history

    