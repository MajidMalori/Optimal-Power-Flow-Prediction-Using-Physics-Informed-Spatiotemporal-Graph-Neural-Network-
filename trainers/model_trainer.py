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
        # Mixed precision removed as per user request (RTX 5090)

    def _train_epoch(self, train_loader):
        self.model.train()
        
        # Get gradient accumulation steps from config (default: 1 = no accumulation)
        grad_accum_steps = getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        
        # Track components
        epoch_metrics = {
            'total_loss': 0.0,
            'mse': 0.0,
            'mae': 0.0,  # Mean Absolute Error (used in plotting)
            'physics_loss': 0.0,
            'safety_loss': 0.0,
            'grad_norm': 0.0  # Track gradient norm for overfitting analysis
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [Train]")
        
        batch_count = 0
        self.optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(pbar):
            # Batch device transfer for efficiency
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            ybus = batch['ybus_matrix'].to(self.device, non_blocking=True)
            adj = batch['adjacency'].to(self.device, non_blocking=True)
            
            # Standard Full Precision Training
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
            
            # Scale loss by accumulation steps (for correct gradient averaging)
            loss = loss / grad_accum_steps
            
            # Backward (accumulate gradients)
            loss.backward()
            
            # Update metrics (scale back to original loss for logging)
            scaled_loss = loss_dict['total_loss'].item()  # Original unscaled loss
            epoch_metrics['total_loss'] += scaled_loss
            
            # Update metrics from loss_dict
            epoch_metrics.update({k: epoch_metrics[k] + loss_dict[k] for k in epoch_metrics if k in loss_dict and k != 'grad_norm'})
            
            # Store weights for progress bar and result
            if self.is_physics_informed:
                if 'weights' not in loss_dict:
                    raise KeyError(f"Physics-informed model requires 'weights' in loss_dict. Got keys: {list(loss_dict.keys())}")
                self._last_weights = loss_dict['weights']
            
            # Calculate MAE (Mean Absolute Error) - used in plotting
            if not self.is_physics_informed:
                mae_batch = torch.nn.functional.l1_loss(outputs, targets).item()
                epoch_metrics['mae'] += mae_batch
            
            batch_count += 1
            
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Calculate gradient norm before clipping (for logging)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                epoch_metrics['grad_norm'] += grad_norm.item()
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()  # Zero gradients for next accumulation cycle
            
            # Clear batch from GPU immediately (after all uses)
            del features, targets, ybus, adj, outputs, loss_dict, loss
            
            # Calculate running averages for progress bar (matches log file values)
            # Only update progress bar every few batches to reduce overhead
            if batch_count % max(1, len(train_loader) // 10) == 0 or batch_idx == len(train_loader) - 1:
                avg_mse = epoch_metrics['mse'] / batch_count
                avg_total_loss = epoch_metrics['total_loss'] / batch_count
                
                # MODE-BASED LOGGING: Controlled by config flags
                show_detailed = getattr(self.config, 'SHOW_DETAILED_PROGRESS', False)
                
                if self.is_physics_informed:
                    avg_phys = epoch_metrics['physics_loss'] / batch_count
                    avg_safe = epoch_metrics['safety_loss'] / batch_count
                    
                    if show_detailed:
                        # Detailed view: Include L and w with lower precision to fit terminal
                        # No defensive check: weights MUST exist for physics-informed models
                        weights = self._last_weights
                        desc = f"L={avg_total_loss:.2f} M={avg_mse:.3f} P={avg_phys:.3f} S={avg_safe:.3f} w=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f}]"
                    else:
                        # Minimal view: M, P, S only with high precision (6 decimal places)
                        # We don't save weights to self._last_weights if not in detailed mode to save space/overhead
                        self._last_weights = None
                        desc = f"M={avg_mse:.6f} P={avg_phys:.6f} S={avg_safe:.6f}"
                else:
                    desc = f"MSE: {avg_mse:.6f}"
                
                pbar.set_postfix_str(desc)
        
        # Average Metrics
        num_batches = len(train_loader)
        result = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        # Add weights to result
        if self.is_physics_informed:
            result['weights'] = self._last_weights
        else:
            result['weights'] = None
        
        return result

    def _val_epoch(self, val_loader):
        self.model.eval()
        epoch_metrics = {
            'total_loss': 0.0,
            'mse': 0.0,
            'mae': 0.0,  
            'physics_loss': 0.0,
            'safety_loss': 0.0
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
                
                # Update metrics from loss_dict
                epoch_metrics.update({k: epoch_metrics[k] + loss_dict[k] for k in epoch_metrics if k in loss_dict})
                
                # Store weights for progress bar and result
                if self.is_physics_informed:
                    if 'weights' not in loss_dict:
                        raise KeyError(f"Physics-informed model requires 'weights' in loss_dict. Got keys: {list(loss_dict.keys())}")
                    last_weights = loss_dict['weights']
                
                # Calculate MAE (Mean Absolute Error) - used in plotting
                if not self.is_physics_informed:
                    mae_batch = torch.nn.functional.l1_loss(outputs, targets).item()
                    epoch_metrics['mae'] += mae_batch
                
                batch_count += 1
                
                # Calculate running averages for progress bar
                avg_mse = epoch_metrics['mse'] / batch_count
                avg_total_loss = epoch_metrics['total_loss'] / batch_count
                
                # MODE-BASED LOGGING: Controlled by config flags
                show_detailed = getattr(self.config, 'SHOW_DETAILED_PROGRESS', False)
                
                if self.is_physics_informed:
                    avg_phys = epoch_metrics['physics_loss'] / batch_count
                    avg_safe = epoch_metrics['safety_loss'] / batch_count
                    
                    if show_detailed:
                        # Detailed view: Include L and w with lower precision
                        # No defensive check: weights MUST exist for physics-informed models
                        desc = f"L={avg_total_loss:.2f} M={avg_mse:.3f} P={avg_phys:.3f} S={avg_safe:.3f} w=[{last_weights[0]:.2f},{last_weights[1]:.2f},{last_weights[2]:.2f}]"
                    else:
                        # Clean view: M, P, S only with high precision (6 decimal places)
                        # We don't save weights to last_weights if not in detailed mode
                        last_weights = None
                        desc = f"M={avg_mse:.6f} P={avg_phys:.6f} S={avg_safe:.6f}"
                else:
                    desc = f"MSE: {avg_mse:.6f}"
                
                pbar.set_postfix_str(desc)
                
                # Clear batch from GPU immediately
                del features, targets, ybus, adj, outputs, loss_dict
        
        num_batches = len(val_loader)
        result = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        # Add weights to result
        if self.is_physics_informed:
            result['weights'] = last_weights
        else:
            result['weights'] = None
        
        return result
