# In utils/metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

def relative_mse_loss(outputs: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Calculates Mean Squared Error relative to the magnitude of the target."""
    relative_error = (outputs - targets) / (torch.abs(targets) + epsilon)
    return torch.mean(relative_error**2)

def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Computes standard regression metrics for evaluation purposes."""
    with torch.no_grad():
        # Ensure outputs and targets have the same shape for comparison
        if outputs.dim() != targets.dim():
             if outputs.dim() == 2 and targets.dim() == 3:
                 num_buses = targets.shape[1]
                 num_features = targets.shape[2]
                 outputs = outputs.view(-1, num_buses, num_features)
             else:
                 raise ValueError(f"Shape mismatch: output {outputs.shape}, target {targets.shape}")

        # Flatten for metric calculation
        outputs_flat = outputs.reshape(outputs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)
        
        mse = F.mse_loss(outputs_flat, targets_flat).item()
        mae = F.l1_loss(outputs_flat, targets_flat).item()
        rmse = torch.sqrt(torch.tensor(mse)).item() if mse >= 0 else float('nan')
            
        return {'mse': mse, 'mae': mae, 'rmse': rmse}

class PowerSystemLoss(nn.Module):
    """
    A comprehensive, physics-informed loss function for power system state estimation.
    This version correctly handles per-sample Ybus matrices and time-varying coefficients,
    making it suitable for datasets with mixed scenarios (e.g., different renewable
    fractions or N-1 contingencies).
    """
    def __init__(self, config: object, normalizer, is_gcn: bool = False):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.is_physics_informed = not is_gcn
        self.mse_loss_fn = nn.MSELoss()
        
        self.register_buffer('v_min', torch.tensor(config.V_MIN, dtype=torch.float32))
        self.register_buffer('v_max', torch.tensor(config.V_MAX, dtype=torch.float32))

        self.loss_scale_factor = getattr(config, "LOSS_SCALE_FACTOR", 1.0) if self.is_physics_informed else 0.0

    # --- START CORRECTION: The forward pass now accepts a BATCH of Ybus matrices and coefficients ---
    def forward(self, 
                outputs_norm: torch.Tensor, 
                targets_norm: torch.Tensor, 
                ybus_batch: torch.Tensor, 
                time_carbon_coeffs: torch.Tensor,
                time_energy_coeffs: torch.Tensor) -> torch.Tensor:
        
        # Ensure outputs and targets have the same shape
        if outputs_norm.dim() != targets_norm.dim():
            if outputs_norm.dim() == 2 and targets_norm.dim() == 3:
                 outputs_norm = outputs_norm.view(targets_norm.shape)
            else:
                 raise ValueError(f"Shape mismatch: outputs {outputs_norm.shape}, targets {targets_norm.shape}")
        
        # 1. Data-Driven Loss (Standard MSE)
        data_loss = self.mse_loss_fn(outputs_norm, targets_norm)
        
        # If not physics-informed, we are done.
        if not self.is_physics_informed:
            return data_loss

        # 2. Physics-Informed Penalties
        # Denormalize model output to physical units for physics calculations
        outputs_denorm = self.normalizer.denormalize(outputs_norm, self.config.NUM_BUSES)
        
        # Calculate physics penalties using the per-sample Ybus matrix
        power_balance_penalty = self._compute_power_balance_violation(outputs_denorm, ybus_batch).mean()
        voltage_limit_penalty = self._compute_voltage_limit_violation(outputs_denorm).mean()
        
        # --- NOTE: The 'carbon emission' penalty is now part of the MOOPF evaluation, not the training loss.
        # This simplifies the training objective to focus on state estimation accuracy and physical validity.
        # The MOOPF objectives (including emissions) are calculated post-training on the best model.
        
        physics_loss = power_balance_penalty + voltage_limit_penalty
        
        # Combine data loss and physics loss
        total_loss = data_loss + self.loss_scale_factor * physics_loss
            
        return total_loss
    # --- END CORRECTION ---

    def _get_power_injections(self, state: torch.Tensor):
        """Helper to extract power values from the state tensor."""
        P_inj = state[:, :, 4] - state[:, :, 2]  # P_gen - P_load
        Q_inj = state[:, :, 5] - state[:, :, 3]  # Q_gen - Q_load
        return state[:, :, 0], state[:, :, 1], P_inj, Q_inj, state[:, :, 2], state[:, :, 3]

    # --- START CORRECTION: This function now works with a batch of Ybus matrices ---
    def _compute_power_balance_violation(self, state: torch.Tensor, ybus_batch: torch.Tensor) -> torch.Tensor:
        """Calculates power mismatch using a batch of Ybus matrices."""
        Vm, Va, P_inj, Q_inj, _, _ = self._get_power_injections(state)
        V = Vm * torch.exp(1j * Va)
        
        # Einsum performs batch-wise matrix-vector multiplication
        # 'bij,bj->bi' means for each item 'b' in the batch, multiply matrix Ybus[b] by vector V[b]
        I = torch.einsum('bij,bj->bi', ybus_batch.cfloat(), V)
        S_calc = V * torch.conj(I)
        
        p_violation = P_inj - S_calc.real
        q_violation = Q_inj - S_calc.imag
        return torch.mean(p_violation**2 + q_violation**2, dim=1)
    # --- END CORRECTION ---

    def _compute_voltage_limit_violation(self, state: torch.Tensor) -> torch.Tensor:
        """Calculates penalty for voltage deviations outside of V_MIN and V_MAX."""
        Vm = state[:, :, 0]
        v_violations = torch.relu(self.v_min - Vm) + torch.relu(Vm - self.v_max)
        return torch.mean(v_violations**2, dim=1)

    # --- The functions below are for post-training MOOPF evaluation, not for the training loss ---

    def _compute_normalized_power_balance_violation(self, state: torch.Tensor, Ybus: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """Computes power balance violation, normalized by the total load for MOOPF evaluation."""
        Vm, Va, P_inj, Q_inj, P_load, Q_load = self._get_power_injections(state)
        V = Vm * torch.exp(1j * Va)
        I = torch.einsum('bij,bj->bi', Ybus.cfloat(), V)
        S_calc = V * torch.conj(I)
        squared_mismatch = (P_inj - S_calc.real)**2 + (Q_inj - S_calc.imag)**2
        total_load_s_squared = torch.sum(P_load**2 + Q_load**2, dim=1)
        return torch.mean(squared_mismatch, dim=1) / (total_load_s_squared + epsilon)

    def _compute_normalized_voltage_limit_violation(self, state: torch.Tensor) -> torch.Tensor:
        """Computes the mean absolute voltage violation in per unit (p.u.) for MOOPF evaluation."""
        Vm = state[:, :, 0]
        v_violations = torch.relu(self.v_min - Vm) + torch.relu(Vm - self.v_max)
        return torch.mean(v_violations, dim=1)

    def _compute_carbon_emissions(
        self, 
        predicted_state_physical: torch.Tensor, 
        time_carbon_coeff: torch.Tensor, 
        time_energy_coeff: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Computes raw and normalized carbon emissions for MOOPF evaluation."""
        total_load = torch.sum(predicted_state_physical[:, :, 2], dim=1)
        total_distributed_gen = torch.sum(predicted_state_physical[:, :, 4], dim=1)
        power_from_grid = torch.relu(total_load - total_distributed_gen)
        
        # Ensure coeffs are correctly shaped for broadcasting
        carbon_intensity = time_carbon_coeff.squeeze(-1) if time_carbon_coeff.dim() > 1 else time_carbon_coeff
        energy_coefficient = time_energy_coeff.squeeze(-1) if time_energy_coeff.dim() > 1 else time_energy_coeff
        
        raw_emissions = (power_from_grid * carbon_intensity) / (energy_coefficient + 1e-9)
        normalized_emissions = power_from_grid / (total_load + 1e-9)
        return {'raw': raw_emissions, 'normalized': normalized_emissions}