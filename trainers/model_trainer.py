from datetime import datetime
import os
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

class PowerSystemTrainer:
    """Trainer for Power System Denoising State Estimator."""
    def __init__(self, model, criterion, optimizer, config, device, is_physics_informed=True):
        self.model = model; self.criterion = criterion; self.optimizer = optimizer
        self.config = config; self.device = device; self.is_physics_informed = is_physics_informed
        self.history = {k: [] for k in ['train_total_loss', 'train_mse', 'train_mae', 'train_physics_loss', 'train_safety_loss', 
                                        'val_total_loss', 'val_mse', 'val_mae', 'val_physics_loss', 'val_safety_loss', 'learning_rate']}
        self.history['train_weights'] = []; self.history['train_log_vars'] = []
        self.best_val_loss = float('inf'); self.epochs_no_improve = 0; self.best_epoch = 0; self.log_file = None
        self.model_name = None; self.num_buses = None

    def _run_epoch(self, loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        metrics = {k: 0.0 for k in ['total_loss', 'mse', 'mae', 'physics_loss', 'safety_loss', 'grad_norm']}
        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS} [{'Train' if is_train else 'Val'}]")
        grad_accum = getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1) if is_train else 1
        last_weights = None
        
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for i, batch in enumerate(pbar):
                feats, targets = batch['features'].to(self.device), batch['targets'].to(self.device)
                ybus, adj = batch['ybus_matrix'].to(self.device), batch['adjacency'].to(self.device)
                
                if is_train and i % grad_accum == 0: self.optimizer.zero_grad()
                
                out = self.model(feats, adj)
                loss_dict = self.criterion(out, targets, feats, ybus, return_components=True, epoch=self.current_epoch)
                loss = loss_dict['total_loss']
                
                if is_train:
                    (loss / grad_accum).backward()
                    if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
                        metrics['grad_norm'] += torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
                        self.optimizer.step()

                for k, v in loss_dict.items(): 
                    if k in metrics: metrics[k] += v.item() if isinstance(v, torch.Tensor) else v
                if not self.is_physics_informed: metrics['mae'] += torch.nn.functional.l1_loss(out, targets).item()
                if self.is_physics_informed: last_weights = loss_dict.get('weights')

                desc = f"M={metrics['mse']/(i+1):.6f}"
                if self.is_physics_informed: desc += f" P={metrics['physics_loss']/(i+1):.6f} S={metrics['safety_loss']/(i+1):.6f}"
                pbar.set_postfix_str(desc)

        if len(loader) == 0:
            return {k: 0.0 for k in metrics}
            
        res = {k: v / len(loader) for k, v in metrics.items()}
        res['weights'] = last_weights
        return res

    def train(self, train_loader, val_loader, model_name, num_buses, config_params=None):
        self.model_name = model_name; self.num_buses = num_buses
        if getattr(self.config, 'LOGGING_ENABLED', False):
            log_path = self.config.get_training_log_path(num_buses, model_name)
            self.log_file = open(log_path, 'a', encoding='utf-8')
            self.log_file.write(f"\n{'#'*80}\n# Run Started: {datetime.now()}\n{'#'*80}\n")
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.NUM_EPOCHS, eta_min=1e-5) if getattr(self.config, 'USE_LEARNING_RATE_SCHEDULER', True) else None
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            self.current_epoch = epoch
            tm = self._run_epoch(train_loader, True)
            vm = self._run_epoch(val_loader, False)
            
            for k, v in tm.items():
                hist_key = k.replace('total_loss', 'train_total_loss') if k=='total_loss' else f'train_{k}'
                if hist_key in self.history: self.history[hist_key].append(v)
            for k, v in vm.items():
                hist_key = k.replace('total_loss', 'val_total_loss') if k=='total_loss' else f'val_{k}'
                if hist_key in self.history: self.history[hist_key].append(v)
            
            self.history['train_weights'].append(tm.get('weights'))
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            if hasattr(self.criterion, 'log_vars'): self.history['train_log_vars'].append([v.item() for v in self.criterion.log_vars])
            else: self.history['train_log_vars'].append(None)

            if scheduler: scheduler.step()
            
            msg = f"Epoch {epoch} | Train: MSE={tm['mse']:.6f} | Val: MSE={vm['mse']:.6f} | Gap={vm['mse']-tm['mse']:+.6f}"
            if self.is_physics_informed: msg += f" | Phys={tm['physics_loss']:.6f} | w={tm.get('weights')}"
            if self.log_file: self.log_file.write(msg + "\n"); self.log_file.flush()
            
            if vm['total_loss'] < self.best_val_loss:
                self.best_val_loss = vm['total_loss']; self.epochs_no_improve = 0; self.best_epoch = epoch
                self._save_checkpoint('best_model.pth')
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                    if self.log_file: self.log_file.write(f"Early stopping at epoch {epoch}. Best: {self.best_epoch}\n")
                    break
                    
        if self.log_file: self.log_file.close()

    def _save_checkpoint(self, filename):
        if not getattr(self.config, 'SAVE_CHECKPOINTS', True): return
        path = os.path.join(self.config.get_model_eval_dir(self.num_buses, self.model_name), filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def get_training_history(self): return self.history
