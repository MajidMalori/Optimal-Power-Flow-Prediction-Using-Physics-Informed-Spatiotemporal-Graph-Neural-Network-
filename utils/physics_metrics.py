import torch
import torch.nn.functional as F
from typing import Dict, Optional
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
    
    def __init__(self, base_mva: float = 100.0):
        self.base_mva = float(base_mva)
        
    def compute_metrics(self, 
                       preds_phys: torch.Tensor, 
                       ybus_batch: torch.Tensor,
                       carbon_intensity: Optional[torch.Tensor] = None,
                       energy_coeff: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute all physics metrics in one pass.
        
        Args:
            preds_phys: Denormalized predictions [batch, buses, 10]
            ybus_batch: Ybus matrices [batch, buses, buses]
            carbon_intensity: Carbon intensity coefficient [batch] (optional)
            energy_coeff: Energy coefficient [batch] (optional)
            
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
        
        # 2. Power Balance / Flow
        # Complex voltage V = vm * exp(j * va)
        V = vm * torch.exp(1j * va)
        
        # Current Injection I = Y * V
        # ybus_batch: [batch, buses, buses] (complex)
        # V: [batch, buses] (complex) -> [batch, buses, 1] for broadcasting
        I = torch.einsum('bij,bj->bi', ybus_batch.cfloat(), V)
        
        # S_flow = V * conj(I) (Calculated Flow)
        S_flow = V * torch.conj(I)
        P_flow = S_flow.real
        Q_flow = S_flow.imag
        
        # Net Injection (Predicted)
        # P_net = (P_ext + P_conv + P_ren) - P_load
        # All inputs are in MW/MVAR, convert to p.u.
        P_gen_mw = p_ext + p_conv + p_ren
        Q_gen_mvar = preds_phys[..., FeatureIndices.Q_EXT_GRID] + \
                     preds_phys[..., FeatureIndices.Q_CONV] + \
                     preds_phys[..., FeatureIndices.Q_REN]
                     
        P_net_pu = (P_gen_mw - p_load) / self.base_mva
        Q_net_pu = (Q_gen_mvar - q_load) / self.base_mva
        
        # Power Mismatch (Balance Violation)
        p_mismatch = F.mse_loss(P_net_pu, P_flow)
        q_mismatch = F.mse_loss(Q_net_pu, Q_flow)
        power_balance_mismatch = p_mismatch + q_mismatch
        
        # 3. Power Loss
        # Real Loss = Sum of Net Injections (Generation - Load)
        # Ideally should be sum(P_flow), which accounts for line losses
        # P_loss_pu = sum(P_flow) across buses
        system_loss_pu = torch.sum(P_flow, dim=1) # [batch]
        # Convert to MW for readability? Keep p.u. for consistency? 
        # Usually % loss is better. Loss / Total Load
        total_load_pu = torch.sum(p_load, dim=1) / self.base_mva
        loss_percentage = torch.mean(torch.abs(system_loss_pu) / (torch.abs(total_load_pu) + 1e-6))
        
        # 4. Carbon Emissions
        # Carbon = (P_conv + max(0, P_ext)) * CarbonIntensity
        # P_conv is always carbon (unless nuclear, but assuming fossil for now or general mix)
        # P_ext: Imports (positive) are carbon, Exports (negative) are saved elsewhere (or 0)
        
        # Carbon emitting generation in MW
        carbon_gen_mw = p_conv + F.relu(p_ext) 
        total_carbon_gen_mw = torch.sum(carbon_gen_mw, dim=1) # [batch]
        
        if carbon_intensity is not None:
            # If intensity provided (e.g., kgCO2/MWh)
            # Ensure shapes match
            if carbon_intensity.dim() > 1:
                carbon_intensity = carbon_intensity.squeeze()
            
            # Normalize intensity if needed, or compute raw
            # Assuming carbon_intensity is per MWh
            emissions = torch.mean(total_carbon_gen_mw * carbon_intensity)
        else:
            # Proxy: Just sum of carbon generation
            emissions = torch.mean(total_carbon_gen_mw)

        return {
            'voltage_deviation': voltage_deviation.item(),
            'power_balance_mismatch': power_balance_mismatch.item(),
            'system_power_loss': loss_percentage.item(),
            'carbon_emissions': emissions.item()
        }
