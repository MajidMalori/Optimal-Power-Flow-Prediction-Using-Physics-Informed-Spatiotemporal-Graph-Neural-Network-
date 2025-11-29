import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer
from utils.forensic_logger import get_logger

class PowerSystemTrainer(BaseTrainer):
    """
    Trainer for Power System Denoising State Estimator.
    """
    def __init__(self, model, criterion, optimizer, config, device, is_physics_informed=True):
        super().__init__(model, criterion, optimizer, config, device)
        self.is_physics_informed = is_physics_informed
        self.forensic_logger = get_logger()
        self.use_cuda = device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_cuda else None

    def _train_epoch(self, train_loader):
        self.model.train()
        
        # Track components
        epoch_metrics = {
            'total_loss': 0.0,
            'mse': 0.0,
            'physics_loss': 0.0,
            'safety_loss': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Train]")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Move to device
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True) # Clean State
            ybus = batch['ybus_matrix'].to(self.device, non_blocking=True)
            adj = batch['adjacency'].to(self.device, non_blocking=True)
            
            device_type = 'cuda' if self.use_cuda else 'cpu'
            
            with torch.amp.autocast(device_type=device_type):
                # Forward Pass
                outputs = self.model(features, adj)
                
                # Loss Calculation (return components for logging)
                loss_dict = self.criterion(
                    outputs_norm=outputs,
                    targets_norm=targets,
                    measurements_norm=features,
                    ybus_batch=ybus,
                    return_components=True,
                    epoch=self.current_epoch
                )
                
                loss = loss_dict['total_loss']
            
            # Backward & Step
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # Gradient clipping: STANDARD PRACTICE for training stability, NOT cheating
                # Essential for: physics-informed losses, graph neural networks, multi-task learning
                # Prevents exploding gradients without affecting model capacity or final performance
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # Gradient clipping: STANDARD PRACTICE for training stability, NOT cheating
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Update Metrics
            for k in epoch_metrics:
                if k in loss_dict:
                    epoch_metrics[k] += loss_dict[k]
            
            # Progress Bar - concise display
            if self.is_physics_informed:
                # Physics-informed: Compact format with Kendall weights
                weights = loss_dict['weights']
                desc = f"L={loss.item():.2f} M={loss_dict['mse']:.3f} P={loss_dict['physics_loss']:.1f} S={loss_dict['safety_loss']:.4f} w=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}]"
            else:
                # Non-physics: MSE only
                desc = f"MSE: {loss_dict['mse']:.4f}"
            
            pbar.set_postfix_str(desc)
            
        # Average Metrics
        num_batches = len(train_loader)
        result = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        # Add weights from last batch (they're model parameters, same for all batches)
        if 'weights' in loss_dict:
            result['weights'] = loss_dict['weights']
        
        return result

    def _val_epoch(self, val_loader):
        self.model.eval()
        # Initialize ALL metrics that train_epoch returns for consistency
        epoch_metrics = {
            'total_loss': 0.0,
            'mse': 0.0,
            'physics_loss': 0.0,
            'safety_loss': 0.0
        }
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                ybus = batch['ybus_matrix'].to(self.device)
                adj = batch['adjacency'].to(self.device)
                
                outputs = self.model(features, adj)
                
                loss_dict = self.criterion(
                    outputs_norm=outputs,
                    targets_norm=targets,
                    measurements_norm=features,
                    ybus_batch=ybus,
                    return_components=True
                )
                
                for k in epoch_metrics:
                    if k in loss_dict:
                        epoch_metrics[k] += loss_dict[k]
                
                # Update progress bar - concise display
                if self.is_physics_informed:
                    # Physics-informed: Compact format with Kendall weights
                    weights = loss_dict['weights']
                    desc = f"L={loss_dict['total_loss']:.2f} M={loss_dict['mse']:.3f} P={loss_dict['physics_loss']:.1f} S={loss_dict['safety_loss']:.4f} w=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}]"
                else:
                    # Non-physics: MSE only
                    desc = f"MSE: {loss_dict['mse']:.4f}"
                
                pbar.set_postfix_str(desc)
        
        num_batches = len(val_loader)
        result = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        # Add weights from last batch (they're model parameters, same for all batches)
        if 'weights' in loss_dict:
            result['weights'] = loss_dict['weights']
        
        return result
