"""
Forensic Debug Logger for Physics-Informed ML Training
========================================================
This module provides comprehensive logging to trace every step of the training pipeline.
Use this to diagnose training issues like model collapse, vanishing gradients, or loss component imbalance.

Usage in config.yaml:
  debug:
    enable: true
    log_dir: "debug_logs"
    log_interval: 10  # Log every N batches
"""

import os
import logging
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

class ForensicLogger:
    """
    FBI-style forensic logger that tracks everything happening in the training pipeline.
    """
    def __init__(self, log_dir="debug_logs", model_name="GCN", bus_system="33", enabled=True, log_interval=10):
        self.enabled = enabled
        if not enabled:
            return
        
        self.log_interval = log_interval  # Store for use by models    
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file - FIXED: Use a single file for the session if it exists
        # We use a fixed filename pattern that doesn't include seconds to allow appending within the same minute/session
        # or we check if a global log file path has been set
        
        if hasattr(ForensicLogger, 'current_log_file') and ForensicLogger.current_log_file:
            log_file = ForensicLogger.current_log_file
            mode = 'a' # Append to existing
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"forensic_{model_name}_{bus_system}bus_{timestamp}.log"
            ForensicLogger.current_log_file = log_file
            mode = 'w'
        
        # Setup logging
        self.logger = logging.getLogger(f"Forensic_{model_name}_{bus_system}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        self.logger.propagate = False # CRITICAL: Prevent propagation to root logger (console)
        
        # File handler
        fh = logging.FileHandler(log_file, mode=mode)
        fh.setLevel(logging.DEBUG)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.step_count = 0
        self.epoch = 0
        
        self.logger.info("="*80)
        self.logger.info(f"FORENSIC DEBUGGING SESSION STARTED")
        self.logger.info(f"Model: {model_name} | Bus System: {bus_system}")
        self.logger.info("="*80)
    
    def log_tensor_stats(self, name, tensor, indent=0):
        """Log detailed statistics for a tensor."""
        if not self.enabled or tensor is None:
            return
            
        prefix = "  " * indent
        
        if isinstance(tensor, torch.Tensor):
            t = tensor.detach().cpu()
            self.logger.debug(f"{prefix}{name}:")
            self.logger.debug(f"{prefix}  Shape: {list(t.shape)}")
            self.logger.debug(f"{prefix}  Range: [{t.min().item():.6f}, {t.max().item():.6f}]")
            self.logger.debug(f"{prefix}  Mean: {t.mean().item():.6f} | Std: {t.std().item():.6f}")
            self.logger.debug(f"{prefix}  NaN: {torch.isnan(t).any().item()} | Inf: {torch.isinf(t).any().item()}")
            
            # Check for dead neurons (all zeros)
            if t.numel() > 0:
                zero_fraction = (t.abs() < 1e-8).float().mean().item()
                self.logger.debug(f"{prefix}  Zero fraction: {zero_fraction:.2%}")
        elif isinstance(tensor, (int, float)):
            self.logger.debug(f"{prefix}{name}: {tensor:.6f}")
        else:
            self.logger.debug(f"{prefix}{name}: {tensor}")
    
    def log_data_batch(self, batch, batch_idx):
        """Log statistics for a data batch."""
        if not self.enabled:
            return
            
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"DATA BATCH {batch_idx}")
        self.logger.info(f"{'='*60}")
        
        self.log_tensor_stats("Features", batch.get('features'), indent=1)
        self.log_tensor_stats("Targets", batch.get('targets'), indent=1)
        self.log_tensor_stats("Adjacency", batch.get('adjacency'), indent=1)
        self.log_tensor_stats("Ybus (real)", batch.get('ybus_matrix').real if batch.get('ybus_matrix') is not None else None, indent=1)
        
        if batch.get('bus_types') is not None:
            bus_types = batch['bus_types']
            self.logger.debug(f"  Bus Types:")
            self.logger.debug(f"    PQ buses (0): {(bus_types == 0).sum().item()}")
            self.logger.debug(f"    PV buses (1): {(bus_types == 1).sum().item()}")
            self.logger.debug(f"    Slack buses (2): {(bus_types == 2).sum().item()}")
    
    def log_model_forward(self, model_name, inputs, outputs):
        """Log model forward pass."""
        if not self.enabled:
            return
            
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"MODEL FORWARD PASS ({model_name})")
        self.logger.info(f"{'='*60}")
        
        self.logger.debug(f"  INPUT:")
        self.log_tensor_stats("Features", inputs.get('features'), indent=2)
        
        self.logger.debug(f"\n  OUTPUT:")
        self.log_tensor_stats("Raw outputs", outputs, indent=2)
        
        # Check for model collapse (outputs constant)
        if isinstance(outputs, torch.Tensor):
            out_std = outputs.std().item()
            if out_std < 1e-6:
                self.logger.warning(f"  ⚠️  MODEL COLLAPSE DETECTED: Output std = {out_std:.2e}")
    
    def log_model_weights(self, model, epoch):
        """Log model weight statistics."""
        if not self.enabled:
            return
            
        self.epoch = epoch
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"MODEL WEIGHTS (Epoch {epoch})")
        self.logger.info(f"{'='*60}")
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.log_tensor_stats(name, param.data, indent=1)
                
                # Log gradients if available
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    self.logger.debug(f"    Gradient norm: {grad_norm:.6f}")
                    
                    if grad_norm < 1e-8:
                        self.logger.warning(f"    ⚠️  VANISHING GRADIENT: {name}")
                    elif grad_norm > 100:
                        self.logger.warning(f"    ⚠️  EXPLODING GRADIENT: {name}")
    
    def log_loss_components(self, loss_dict, step, phase="train"):
        """Log detailed loss component breakdown."""
        if not self.enabled:
            return
            
        self.step_count = step
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"LOSS BREAKDOWN ({phase.upper()} - Step {step})")
        self.logger.info(f"{'='*60}")
        
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                val = value.item()
            else:
                val = value
            self.logger.info(f"  {key:30s}: {val:.6f}")
        
        # Check for imbalanced loss components
        if 'total_loss' in loss_dict:
            total = loss_dict['total_loss'].item() if isinstance(loss_dict['total_loss'], torch.Tensor) else loss_dict['total_loss']
            
            for key in ['mse', 'power_violation', 'voltage_violation']:
                if key in loss_dict:
                    component = loss_dict[key].item() if isinstance(loss_dict[key], torch.Tensor) else loss_dict[key]
                    if total > 0:
                        ratio = component / total
                        self.logger.debug(f"  {key} / total_loss: {ratio:.2%}")
                        
                        if key == 'mse' and ratio < 0.01:
                            self.logger.warning(f"  ⚠️  MSE is < 1% of total loss - physics constraints may be dominating")
    
    def log_gradient_flow(self, model):
        """Log gradient flow through the model."""
        if not self.enabled:
            return
            
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"GRADIENT FLOW")
        self.logger.info(f"{'='*60}")
        
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                self.logger.debug(f"  {name:40s}: {param_norm:.6f}")
        
        total_norm = total_norm ** 0.5
        self.logger.info(f"  Total Gradient Norm: {total_norm:.6f}")
        
        if total_norm < 1e-6:
            self.logger.error(f"  🚨 CRITICAL: Gradient vanishing (norm = {total_norm:.2e})")
        elif total_norm > 1000:
            self.logger.error(f"  🚨 CRITICAL: Gradient exploding (norm = {total_norm:.2e})")
    
    def log_optimization_step(self, optimizer, lr):
        """Log optimizer state after update."""
        if not self.enabled:
            return
            
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"OPTIMIZATION STEP {self.step_count}")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  Learning Rate: {lr:.6e}")
        
        # Log optimizer state (e.g., momentum for Adam)
        for i, param_group in enumerate(optimizer.param_groups):
            self.logger.debug(f"  Param Group {i}:")
            self.logger.debug(f"    LR: {param_group['lr']:.6e}")
            if 'weight_decay' in param_group:
                self.logger.debug(f"    Weight Decay: {param_group['weight_decay']:.6e}")
    
    def log_epoch_summary(self, epoch, train_metrics, val_metrics):
        """Log end-of-epoch summary."""
        if not self.enabled:
            return
            
        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"EPOCH {epoch} SUMMARY")
        self.logger.info(f"{'#'*80}")
        
        self.logger.info(f"\n  TRAIN:")
        for key, value in train_metrics.items():
            self.logger.info(f"    {key:25s}: {value:.6f}")
        
        self.logger.info(f"\n  VALIDATION:")
        for key, value in val_metrics.items():
            self.logger.info(f"    {key:25s}: {value:.6f}")
        
        # Check for overfitting
        if 'mse' in train_metrics and 'mse' in val_metrics:
            train_mse = train_metrics['mse']
            val_mse = val_metrics['mse']
            if val_mse > train_mse * 1.5:
                self.logger.warning(f"  ⚠️  Potential overfitting: Val MSE {val_mse/train_mse:.2f}x Train MSE")
        
        self.logger.info(f"{'#'*80}\n")
    
    def log_diagnosis(self, message):
        """Log a diagnostic message or finding."""
        if not self.enabled:
            return
        self.logger.warning(f"\n🔍 DIAGNOSIS: {message}")
    
    def close(self):
        """Close the logger."""
        if not self.enabled:
            return
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"FORENSIC DEBUGGING SESSION ENDED")
        self.logger.info(f"{'='*80}")
        
        for handler in self.logger.handlers:
            handler.close()


# Global logger instance (will be initialized by train.py)
_global_logger = None

def init_forensic_logger(log_dir="debug_logs", model_name="GCN", bus_system="33", enabled=True, log_interval=10):
    """Initialize the global forensic logger."""
    global _global_logger
    _global_logger = ForensicLogger(log_dir, model_name, bus_system, enabled, log_interval)
    return _global_logger

def get_logger():
    """Get the global forensic logger."""
    return _global_logger

def close_logger():
    """Close the global forensic logger."""
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None
