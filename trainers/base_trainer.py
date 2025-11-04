# In trainers/base_trainer.py

import torch
from abc import ABC, abstractmethod

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
        
        # ETH Zurich Technique 1: ReduceLROnPlateau Scheduler
        # Reduces learning rate by factor of 0.1 when validation loss plateaus for 10 epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',           # Minimize loss
            factor=0.1,           # Reduce LR by 10x
            patience=10,          # Wait 10 epochs before reducing
            threshold=0.01,       # Minimum change to qualify as improvement (1%)
            threshold_mode='rel'  # Relative change
            # Note: 'verbose' parameter removed for PyTorch <1.9 compatibility
            # LR changes will still be tracked in history
        )
        
        # Initialize attributes for tracking training progress
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.best_epoch = 0  # Track best epoch for improved early stopping

        # Add history tracking
        self.history = {
            'train_total_loss': [], 'train_mse': [], 
            'train_mse_vm': [], 'train_mse_va': [],  # ETH Zurich: Separate VM/VA tracking
            'train_power_violation': [], 'train_voltage_violation': [],
            'val_total_loss': [], 'val_mse': [], 
            'val_mse_vm': [], 'val_mse_va': [],  # ETH Zurich: Separate VM/VA tracking
            'val_power_violation': [], 'val_voltage_violation': [],
            'learning_rates': [],  # Track LR changes
            # Learnable Uncertainty Weighting (Kendall et al., CVPR 2018)
            'sigma_data': [], 'sigma_power': [], 'sigma_voltage': [],
            'effective_lambda_p': [], 'effective_lambda_v': []  # Effective weights = 1/(2σ²)
        }

    @abstractmethod
    def _train_epoch(self, train_loader):
        """Logic for a single training epoch. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def _val_epoch(self, val_loader):
        """Logic for a single validation epoch. Must be implemented by subclasses."""
        raise NotImplementedError

    # --- START CORRECTION: The main training loop is now correctly implemented here ---
    def train(self, train_loader, val_loader):
        """
        Main training loop. This function iterates over epochs, calls the
        training and validation epoch methods, and handles early stopping.
        """
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            # This is the crucial line that sets the attribute for the tqdm progress bar
            self.current_epoch = epoch
            
            # Call the specific implementation for one training epoch
            train_metrics = self._train_epoch(train_loader)

            # Store training metrics
            self.history['train_total_loss'].append(train_metrics['loss'])
            self.history['train_mse'].append(train_metrics['mse'])
            self.history['train_mse_vm'].append(train_metrics.get('mse_vm', 0.0))  # ETH Zurich
            self.history['train_mse_va'].append(train_metrics.get('mse_va', 0.0))  # ETH Zurich
            self.history['train_power_violation'].append(train_metrics['power_violation'])
            self.history['train_voltage_violation'].append(train_metrics['voltage_violation'])
            
            # Call the specific implementation for one validation epoch
            val_metrics = self._val_epoch(val_loader)

            # Store validation metrics
            self.history['val_total_loss'].append(val_metrics['loss'])
            self.history['val_mse'].append(val_metrics['mse'])
            self.history['val_mse_vm'].append(val_metrics.get('mse_vm', 0.0))  # ETH Zurich
            self.history['val_mse_va'].append(val_metrics.get('mse_va', 0.0))  # ETH Zurich
            self.history['val_power_violation'].append(val_metrics['power_violation'])
            self.history['val_voltage_violation'].append(val_metrics['voltage_violation'])
            
            # Print learnable uncertainty parameters (Kendall et al., CVPR 2018)
            if hasattr(self, 'criterion') and hasattr(self.criterion, 'log_sigma_data'):
                sigma_data = torch.exp(self.criterion.log_sigma_data).item()
                sigma_power = torch.exp(self.criterion.log_sigma_power).item()
                sigma_voltage = torch.exp(self.criterion.log_sigma_voltage).item()
                effective_lambda_p = 1.0 / (2.0 * sigma_power ** 2)
                effective_lambda_v = 1.0 / (2.0 * sigma_voltage ** 2)
                
                # Track in history for plotting
                self.history['sigma_data'].append(sigma_data)
                self.history['sigma_power'].append(sigma_power)
                self.history['sigma_voltage'].append(sigma_voltage)
                self.history['effective_lambda_p'].append(effective_lambda_p)
                self.history['effective_lambda_v'].append(effective_lambda_v)
                
                print(f"  Learnable σ (data, power, voltage): ({sigma_data:.4f}, {sigma_power:.4f}, {sigma_voltage:.4f})")
                print(f"  Effective λ (power, voltage): ({effective_lambda_p:.4f}, {effective_lambda_v:.4f})")
                # Note: total_loss uses weighted components, so it can be < raw MSE (this is expected)
                # total = (1/(2σ²)) * MSE + (1/(2σ²)) * power + (1/(2σ²)) * voltage + log(σ) terms

            # # Display epoch summary only for physics-informed models
            # if hasattr(self, 'is_physics_informed') and self.is_physics_informed:
            #     print(f"Epoch {epoch} Summary:")
            #     print(f"  Train Loss: {train_metrics['loss']:.6f} (MSE: {train_metrics['mse']:.6f} + Physics: {train_metrics['power_violation']:.6f} + {train_metrics['voltage_violation']:.6f})")
            #     print(f"  Val Loss:   {val_metrics['loss']:.6f} (MSE: {val_metrics['mse']:.6f} + Physics: {val_metrics['power_violation']:.6f} + {val_metrics['voltage_violation']:.6f})")

            # ETH Zurich Technique 1: Step the learning rate scheduler
            val_loss = val_metrics.get('loss', float('inf'))
            
            # Track old LR before stepping
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)  # Reduce LR if loss plateaus
            
            # Track current learning rate and print if changed
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Manual verbose notification (for PyTorch <1.9 compatibility)
            if current_lr != old_lr:
                print(f"  ReduceLROnPlateau: Learning rate reduced to {current_lr:.2e}")
            
            # ETH Zurich Technique 3: Improved early stopping with relative improvement threshold
            # Stop if no relative improvement > 1% for patience epochs
            # For physics-informed models, also consider physics violations
            
            # Check if this is a significant improvement (>1% reduction)
            if epoch > 1:
                relative_improvement = (self.best_val_loss - val_loss) / (self.best_val_loss + 1e-10)
            else:
                relative_improvement = 0.01  # First epoch always counts as improvement
            
            if val_loss < self.best_val_loss:
                # Track if improvement is significant (ETH Zurich uses 1% threshold)
                if relative_improvement > 0.01:
                    self.best_epoch = epoch
                    
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                # Save the best model checkpoint
                self._save_checkpoint('best_model.pth')
            else:
                self.epochs_no_improve += 1
                
                # ETH Zurich condition: No significant improvement for many epochs
                epochs_since_significant = epoch - self.best_epoch
                if epochs_since_significant >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping: No >1% improvement for {epochs_since_significant} epochs (best at epoch {self.best_epoch}).")
                    break
                elif self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch} epochs.")
                    break
    # --- END CORRECTION ---

    def _save_checkpoint(self, filename):
        """Helper function to save model checkpoints."""
        # This assumes your config object has a method to get checkpoint paths
        if hasattr(self.config, 'get_checkpoint_path'):
            path = self.config.get_checkpoint_path(filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)

    def get_training_history(self):
        """Return the training history dictionary."""
        return self.history

    