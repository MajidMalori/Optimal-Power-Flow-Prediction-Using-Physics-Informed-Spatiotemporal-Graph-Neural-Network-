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
            'mae': 0.0,  # Mean Absolute Error (for non-physics models)
            'physics_loss': 0.0,
            'safety_loss': 0.0,
            'constraint_loss': 0.0,
            'grad_norm': 0.0  # Track gradient norm for overfitting analysis
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Train]")
        
        batch_count = 0
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Use common processing
            # Note: We need to handle autocast/backward explicitly for training
            # So we can't fully reuse _process_batch if it includes the context manager 
            # unless we pass a flag or handle it carefully.
            # Let's duplicate the data moving parts but keep forward/loss separate? 
            # Or just duplicate the context manager.
            
            # Batch device transfer for efficiency
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            ybus = batch['ybus_matrix'].to(self.device, non_blocking=True)
            adj = batch['adjacency'].to(self.device, non_blocking=True)
            
            device_type = 'cuda' if self.use_cuda else 'cpu'
            
            with torch.amp.autocast(device_type=device_type):
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
            
            # Backward & Step
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # Calculate gradient norm before clipping (for logging)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                epoch_metrics['grad_norm'] += grad_norm.item()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # Calculate gradient norm before clipping (for logging)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                epoch_metrics['grad_norm'] += grad_norm.item()
                self.optimizer.step()
            
            # Update Metrics (vectorized dict comprehension)
            epoch_metrics.update({k: epoch_metrics[k] + loss_dict[k] for k in epoch_metrics if k in loss_dict and k != 'grad_norm'})
            
            # Calculate MAE (Mean Absolute Error) for non-physics models
            if not self.is_physics_informed:
                mae_batch = torch.nn.functional.l1_loss(outputs, targets).item()
                epoch_metrics['mae'] += mae_batch
            
            batch_count += 1
            # Calculate running averages for progress bar (matches log file values)
            avg_mse = epoch_metrics['mse'] / batch_count
            avg_total_loss = epoch_metrics['total_loss'] / batch_count
            
            # Progress Bar - Show running averages (matches epoch-averaged values in log)
            if self.is_physics_informed:
                weights = loss_dict['weights']
                avg_phys = epoch_metrics['physics_loss'] / batch_count
                avg_safe = epoch_metrics['safety_loss'] / batch_count
                avg_constraint = epoch_metrics['constraint_loss'] / batch_count
                desc = f"L={avg_total_loss:.2f} M={avg_mse:.3f} P={avg_phys:.3f} S={avg_safe:.3f} C={avg_constraint:.3f} w=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f},{weights[3]:.2f}]"
            else:
                desc = f"MSE: {avg_mse:.4f}"
            
            pbar.set_postfix_str(desc)
            
        # Average Metrics
        num_batches = len(train_loader)
        result = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        if 'weights' in loss_dict:
            result['weights'] = loss_dict['weights']
        
        return result

    def _val_epoch(self, val_loader):
        self.model.eval()
        epoch_metrics = {
            'total_loss': 0.0,
            'mse': 0.0,
            'mae': 0.0,  # Mean Absolute Error (for non-physics models)
            'physics_loss': 0.0,
            'safety_loss': 0.0,
            'constraint_loss': 0.0
        }
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Val]")
        
        batch_count = 0
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
                
                # Update Metrics (vectorized dict comprehension)
                epoch_metrics.update({k: epoch_metrics[k] + loss_dict[k] for k in epoch_metrics if k in loss_dict})
                
                # Calculate MAE (Mean Absolute Error) for non-physics models
                if not self.is_physics_informed:
                    mae_batch = torch.nn.functional.l1_loss(outputs, targets).item()
                    epoch_metrics['mae'] += mae_batch
                
                batch_count += 1
                # Calculate running averages for progress bar (matches log file values)
                avg_mse = epoch_metrics['mse'] / batch_count
                avg_total_loss = epoch_metrics['total_loss'] / batch_count
                
                # Progress Bar - Show running averages (matches epoch-averaged values in log)
                if self.is_physics_informed:
                    weights = loss_dict['weights']
                    avg_phys = epoch_metrics['physics_loss'] / batch_count
                    avg_safe = epoch_metrics['safety_loss'] / batch_count
                    avg_constraint = epoch_metrics['constraint_loss'] / batch_count
                    desc = f"L={avg_total_loss:.2f} M={avg_mse:.3f} P={avg_phys:.3f} S={avg_safe:.3f} C={avg_constraint:.3f} w=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f},{weights[3]:.2f}]"
                else:
                    desc = f"MSE: {avg_mse:.4f}"
                
                pbar.set_postfix_str(desc)
        
        num_batches = len(val_loader)
        result = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        if 'weights' in loss_dict:
            result['weights'] = loss_dict['weights']
        
        return result
