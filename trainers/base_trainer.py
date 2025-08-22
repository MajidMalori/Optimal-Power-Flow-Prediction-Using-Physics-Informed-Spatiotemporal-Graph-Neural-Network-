# In trainers/base_trainer.py

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
        
        # Initialize attributes for tracking training progress
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

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
            
            # Call the specific implementation for one validation epoch
            val_metrics = self._val_epoch(val_loader)

            print(f"Epoch {epoch}/{self.config.NUM_EPOCHS} | "
                  f"Train Loss: {train_metrics.get('loss', float('nan')):.4f} | "
                  f"Val Loss: {val_metrics.get('loss', float('nan')):.4f}")

            # Early stopping logic
            val_loss = val_metrics.get('loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                # Save the best model checkpoint
                self._save_checkpoint('best_model.pth')
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
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
            print(f"Checkpoint saved to {path}")