import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from config import FeatureIndices

class PowerSystemLoss(nn.Module):
    """
    Physics-Informed Denoising Loss for Power System State Estimation.
    
    Loss Components:
    1. L1 (Data Accuracy): MSE(predictions, clean_targets) - all 10 features.
    2. L2 (Physics Balance): MSE(Net Injection - Flow).
    3. L3 (Safety): Voltage limit violations.
    
    Weights are learned using Kendall's Loss (Multi-Task Learning Using Uncertainty).
    """
    def __init__(self, config: object, normalizer, is_gcn: bool = False):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.is_physics_informed = not is_gcn
        self.mse_loss_fn = nn.MSELoss()
        
        # Learnable weights (log variance) for Kendall's Loss
        # Initialize with 0.0 (variance = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(3))  # [Data, Physics, Safety]
        
        # System Limits
        self.register_buffer('v_min', torch.tensor(getattr(config, 'V_MIN', 0.9), dtype=torch.float32))
        self.register_buffer('v_max', torch.tensor(getattr(config, 'V_MAX', 1.1), dtype=torch.float32))
        
        # Base MVA for unit conversion (if needed internally, though we work in p.u. mostly)
        self.base_mva = getattr(normalizer, 'base_mva', 100.0)

    def forward(self, 
                outputs_norm: torch.Tensor,
                targets_norm: torch.Tensor,
                measurements_norm: torch.Tensor, # Not used in Denoising (we compare preds vs clean targets)
                ybus_batch: torch.Tensor,
                bus_types: torch.Tensor = None,
                return_components: bool = False,
                epoch: int = None) -> torch.Tensor:
        """
        Forward pass for loss calculation.
        
        Args:
            outputs_norm: Predicted clean state [batch, buses, 10] (Normalized)
            targets_norm: True clean state [batch, buses, 10] (Normalized)
            ybus_batch: Ybus matrices [batch, buses, buses] (p.u.)
        """
        # 1. Data Loss (L1) - Compare Normalized Predictions to Normalized Targets
        # This ensures equal weighting across features regardless of scale
        l1_loss = F.mse_loss(outputs_norm, targets_norm)
        
        if not self.is_physics_informed:
            if return_components:
                return {
                    'total_loss': l1_loss,
                    'mse': l1_loss.item(),
                    'physics': 0.0,
                    'safety': 0.0
                }
            return l1_loss

        # 2. Physics Loss (L2) - Power Balance
        # Requires Denormalization to Physical Units (p.u. for Voltage, MW/MVar for Power)
        # Or better: Convert everything to per-unit for physics calculation.
        # Our Global Per-Unit scaling in Normalizer handles this!
        # Actually, Normalizer: Power -> val / base_mva. So normalized power IS per-unit power.
        # Normalizer: Voltage -> (val - 1.0) * 10.0. We need to reverse this to get p.u. Voltage.
        
        # Denormalize ONLY for Physics Calculation (to get V in p.u. and P/Q in p.u.)
        # Since our "Normalized" Power is ALREADY p.u. (Power/BaseMVA), we can use it directly?
        # No, let's stick to explicit denormalization to be safe and consistent.
        # Wait, denormalize returns physical units (MW/MVar).
        # We need p.u. for Ybus calculation.
        # So: Denormalize -> Divide Power by BaseMVA -> Physics Check.
        
        preds_phys = self.normalizer.denormalize(outputs_norm)
        
        # Extract State Variables (Physical Units)
        # Voltage Magnitude (p.u.)
        vm_pu = preds_phys[..., FeatureIndices.VM]
        # Voltage Angle (radians)
        va_rad = preds_phys[..., FeatureIndices.VA]
        
        # Construct Complex Voltage
        V = vm_pu * torch.exp(1j * va_rad)
        
        # Calculate Theoretical Power Flow (Injection)
        # S = V * conj(Y * V)
        # Ybus is in p.u.
        I_inj = torch.einsum('bij,bj->bi', ybus_batch.cfloat(), V)
        S_flow_pu = V * torch.conj(I_inj)
        P_flow_pu = S_flow_pu.real
        Q_flow_pu = S_flow_pu.imag
        
        # Extract Predicted Net Injections (Physical Units -> Convert to p.u.)
        # P_net = (P_ext + P_conv + P_ren) - P_load
        p_load = preds_phys[..., FeatureIndices.P_LOAD]
        p_ext = preds_phys[..., FeatureIndices.P_EXT_GRID]
        p_conv = preds_phys[..., FeatureIndices.P_CONV]
        p_ren = preds_phys[..., FeatureIndices.P_REN]
        
        q_load = preds_phys[..., FeatureIndices.Q_LOAD]
        q_ext = preds_phys[..., FeatureIndices.Q_EXT_GRID]
        q_conv = preds_phys[..., FeatureIndices.Q_CONV]
        q_ren = preds_phys[..., FeatureIndices.Q_REN]
        
        # Calculate Net Injection (MW/MVar)
        P_net_mw = (p_ext + p_conv + p_ren) - p_load
        Q_net_mvar = (q_ext + q_conv + q_ren) - q_load
        
        # Convert to p.u.
        P_net_pu = P_net_mw / self.base_mva
        Q_net_pu = Q_net_mvar / self.base_mva
        
        # L2 Loss: MSE of Mismatch
        l2_loss = F.mse_loss(P_net_pu, P_flow_pu) + F.mse_loss(Q_net_pu, Q_flow_pu)
        
        # 3. Safety Loss (L3) - Voltage Limits
        # Soft constraints on Voltage Magnitude
        # ReLU(|V| - 1.1)^2 + ReLU(0.9 - |V|)^2
        v_upper_violation = F.relu(vm_pu - self.v_max)
        v_lower_violation = F.relu(self.v_min - vm_pu)
        l3_loss = torch.mean(v_upper_violation**2 + v_lower_violation**2)
        
        # 4. Combine with Kendall's Loss Weighting
        # Loss = Σ (L_i * exp(-s_i) + s_i)
        # s_i = log_var
        
        # Precision weights (exp(-s))
        w_data = torch.exp(-self.log_vars[0])
        w_phys = torch.exp(-self.log_vars[1])
        w_safe = torch.exp(-self.log_vars[2])
        
        # Weighted Loss
        loss = (w_data * l1_loss + self.log_vars[0]) + \
               (w_phys * l2_loss + self.log_vars[1]) + \
               (w_safe * l3_loss + self.log_vars[2])
               
        if return_components:
            return {
                'total_loss': loss,
                'mse': l1_loss.item(),
                'physics_loss': l2_loss.item(),
                'safety_loss': l3_loss.item(),
                'sigmas': torch.exp(0.5 * self.log_vars).detach().cpu().numpy().tolist()
            }
            
        return loss

def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, ybus_batch: torch.Tensor, config: object, bus_types: torch.Tensor) -> Dict[str, float]:
    """
    Computes evaluation metrics.
    Args:
        outputs: Predicted Clean State (Physical Units)
        targets: True Clean State (Physical Units)
    """
    with torch.no_grad():
        # MSE
        mse = F.mse_loss(outputs, targets).item()
        rmse = mse ** 0.5
        
        return {'mse': mse, 'rmse': rmse}
