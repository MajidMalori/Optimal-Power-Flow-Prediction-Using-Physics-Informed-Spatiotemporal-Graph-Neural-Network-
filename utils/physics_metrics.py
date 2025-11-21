"""
Physics Metrics Computation for Robustness Analysis

This module provides functions to compute actual physics violations
(power balance, voltage limits) for model evaluation under contingencies.
"""

import torch
import torch.nn.functional as F
from typing import Dict
from utils.metrics import PowerSystemLoss


def compute_physics_metrics(predictions: torch.Tensor, measurements: torch.Tensor, 
                           ybus_batch: torch.Tensor, config: object, 
                           bus_types: torch.Tensor) -> Dict[str, float]:
    """
    Compute actual physics violations (power balance and voltage limits).
    
    This function is specifically designed for robustness analysis where we need
    to compute physics violations for model predictions under contingencies.
    
    Args:
        predictions: Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
        measurements: Measured power [batch, buses, 10] = [p_load, q_load, ..., vm_meas, va_meas]
        ybus_batch: Ybus matrices [batch, buses, buses] (may be modified for contingencies)
        config: Configuration object
        bus_types: Required [batch, buses] with codes [0=PQ, 1=PV, 2=Slack]
    
    Returns:
        Dictionary with 'power_violation' and 'voltage_violation' (RMSE in p.u.)
    """
    if bus_types is None:
        raise ValueError("bus_types is required for physics metrics computation")
    
    # Create a PowerSystemLoss instance to use its physics violation methods
    # We only need the violation computation, not the full loss
    physics_calculator = PowerSystemLoss(config)
    
    with torch.no_grad():
        # Compute power balance violation (RMSE)
        # This uses the same method as in the loss function
        power_violation_rmse = physics_calculator._compute_power_balance_violation(
            predictions, measurements, ybus_batch, bus_types, 
            squared=False  # Return RMSE, not MSE
        )  # [batch] - RMSE per sample
        
        # Compute voltage limit violation (RMSE)
        voltage_violation_rmse = physics_calculator._compute_voltage_limit_violation(
            predictions, measurements, bus_types,
            squared=False  # Return RMSE, not MSE
        )  # [batch] - RMSE per sample
        
        # Take mean across batch to get single metric
        power_violation = torch.mean(power_violation_rmse).item()  # Scalar - mean RMSE (p.u.)
        voltage_violation = torch.mean(voltage_violation_rmse).item()  # Scalar - mean RMSE (p.u.)
    
    return {
        'power_violation': power_violation,
        'voltage_violation': voltage_violation
    }

