import torch
import torch.nn.functional as F
from typing import Dict
from config import FeatureIndices

class PhysicsMetricEngine:
    """
    Highly optimized, vectorized physics metric engine.
    
    Calculates:
    1. Power Flow: S = V * conj(Y * V)
    2. Power Loss: Real part of generation - load
    3. Voltage Deviation: Mean absolute deviation from 1.0 p.u.
    4. Carbon Emissions: Based on generation mix
    """
    
    def __init__(self, base_mva: float):
        self.base_mva = float(base_mva)
        
    def compute_metrics(self, 
                       preds_phys: torch.Tensor, 
                       ybus_batch: torch.Tensor,
                       carbon_intensity: torch.Tensor,
                       energy_coeff: torch.Tensor,
                       renewable_fraction: torch.Tensor) -> Dict[str, float]:
        """
        Compute all physics metrics in one pass.
        
        Args:
            preds_phys: Denormalized predictions [batch, buses, 10]
            ybus_batch: Ybus matrices [batch, buses, buses]
            carbon_intensity: Carbon intensity coefficient [batch] (required, always present)
            energy_coeff: Energy coefficient [batch] (required, always present)
            
        Returns:
            Dictionary of scalar metrics (mean over batch)
        """
        # Extract components
        # [batch, buses]
        vm = preds_phys[..., FeatureIndices.VM]
        va = preds_phys[..., FeatureIndices.VA]
        
        p_load = preds_phys[..., FeatureIndices.P_LOAD]
        q_load = preds_phys[..., FeatureIndices.Q_LOAD]
        p_ext = preds_phys[..., FeatureIndices.P_EXT_GRID]
        p_conv = preds_phys[..., FeatureIndices.P_CONV]
        p_ren = preds_phys[..., FeatureIndices.P_REN]
        
        # 1. Voltage Deviation
        # Mean absolute deviation from 1.0 p.u.
        voltage_deviation = torch.mean(torch.abs(vm - 1.0))
        
        # 2. Power Flow Magnitude (ACOPF equation 3.8)
        # Complex voltage V = vm * exp(j * va)
        V = vm * torch.exp(1j * va)
        
        # Current Injection I = Y * V
        # This implements: I_i = Σ_s Y_is * V_s
        I = torch.einsum('bij,bj->bi', ybus_batch.cfloat(), V)
        
        # Complex power flows: S = V * conj(I)
        # This implements the ACOPF equation (3.8):
        # S_i = V_i * conj(I_i) = V_i * Σ_s Y_is* * V_s*
        # P_i = Re(S_i) = V_i Σ_s V_s (G_is cos θ_is + B_is sin θ_is)
        # Q_i = Im(S_i) = V_i Σ_s V_s (G_is sin θ_is - B_is cos θ_is)
        S_calc_pu = V * torch.conj(I)  # [batch, buses]
        
        # Extract active and reactive power flows (preserve signs for physics accuracy)
        p_flow_values = S_calc_pu.real  # [batch, buses]
        q_flow_values = S_calc_pu.imag  # [batch, buses]
        
        # Calculate apparent power flow magnitude per bus
        s_flow_magnitudes = torch.sqrt(p_flow_values**2 + q_flow_values**2)  # [batch, buses]
        
        # Calculate mean apparent power flow magnitude per bus
        mean_flow_magnitude_per_bus = torch.mean(s_flow_magnitudes, dim=-1)  # [batch]
        
        # Normalize by total load magnitude
        p_load_total = torch.sum(p_load, dim=-1)  # [batch]
        q_load_total = torch.sum(q_load, dim=-1)  # [batch]
        total_load_magnitude = torch.sqrt(p_load_total**2 + q_load_total**2)
        total_load_pu = total_load_magnitude / self.base_mva
        
        power_flow = torch.mean(mean_flow_magnitude_per_bus / (total_load_pu + 1e-9))
        
        # 3. Power Loss
        # Real Loss = Sum of active power flows across buses
        system_loss_pu = torch.sum(p_flow_values, dim=1) / self.base_mva  # [batch]
        # Loss percentage relative to total load
        loss_percentage = torch.mean(torch.abs(system_loss_pu) / (total_load_pu + 1e-6))
        
        # 4. Carbon Emissions (Physics-Consistent with Gemini Approach)
        # CONSISTENCY: Like power_flow and power_loss, calculate from PHYSICS not predictions
        # 
        # Research Paper Formula: f3 = (P_carbon * Cm) / Ef
        # But calculate P_carbon from PHYSICS (V, Y), not from predicted P_conv/P_ren
        
        # Step 1: Calculate TOTAL generation from physics (already have P_flow from above)
        # P_gen = sum of all injections at buses
        # From ACOPF: P_gen_total = sum(P_flow) per bus
        # We already calculated S_calc_pu = V * conj(I) above
        p_gen_per_bus = p_flow_values  # [batch, buses] - Real power from physics
        
        # Total system generation (sum positive generation only)
        p_gen_total = torch.sum(F.relu(p_gen_per_bus), dim=1)  # [batch]
        
        # Step 2: Split into carbon/renewable using renewable_fraction from data
        # Physics-based approach (consistent with power flow/loss)
        # renewable_fraction is always present from data loader (line 303 in data_loader.py)
        
        if renewable_fraction.dim() > 1:
            renewable_fraction = renewable_fraction.squeeze()
        
        # P_carbon = P_total * (1 - renewable_fraction)
        total_carbon_emitting_gen = p_gen_total * (1.0 - renewable_fraction)  # [batch]
        total_generation = p_gen_total  # [batch]
        
        # Ensure carbon_intensity and energy_coeff are 1D
        if carbon_intensity.dim() > 1:
            carbon_intensity = carbon_intensity.squeeze()
        if energy_coeff.dim() > 1:
            energy_coeff = energy_coeff.squeeze()
        
        # Raw emissions using research paper formula
        raw_emissions = (total_carbon_emitting_gen * carbon_intensity) / (energy_coeff + 1e-9)  # [batch]
        
        # Normalize to per-unit by dividing by maximum possible emissions
        # (total_generation * carbon_intensity / energy_coeff)
        # This gives: (carbon_gen * C/E) / (total_gen * C/E) = carbon_gen / total_gen
        max_possible_emissions = (total_generation * carbon_intensity) / (energy_coeff + 1e-9)  # [batch]
        emissions_pu = raw_emissions / (max_possible_emissions + 1e-9)  # [batch]
        
        emissions = torch.mean(emissions_pu)

        return {
            'voltage_deviation': voltage_deviation.item(),
            'power_flow': power_flow.item(),  # For tracking/plotting only, not in MOOPF
            'system_power_loss': loss_percentage.item(),
            'carbon_emissions': emissions.item(),  # Per-unit (for MOOPF)
            'carbon_emissions_raw': torch.mean(raw_emissions).item()  # For tracking/plotting only, not in MOOPF
        }
