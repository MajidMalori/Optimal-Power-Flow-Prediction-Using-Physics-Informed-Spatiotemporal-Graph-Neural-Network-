import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FeatureIndices

class PowerSystemLoss(nn.Module):
    """
    Physics-Informed Denoising Loss for Power System State Estimation.
    
    Loss Components:
    1. L1 (Data Accuracy): MSE(predictions, clean_targets) - all 10 features.
    2. L2 (Physics Balance): MSE(Net Injection - Flow).
    3. L3 (Safety): Voltage limit violations.
    
    Uses Kendall's Homoscedastic Uncertainty method to automatically learn task weights.
    MC Dropout handles aleatoric uncertainty quantification during evaluation.
    """
    def __init__(self, config: object, normalizer, is_gcn: bool = False):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.is_physics_informed = not is_gcn
        
        # Learnable weights using Kendall's method (log variance for numerical stability)
        # Initialize with 0.0 (variance = 1.0, precision weight = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(3))  # [Data, Physics, Safety]
        
        # System Limits
        self.register_buffer('v_min', torch.tensor(getattr(config, 'V_MIN', 0.9), dtype=torch.float32))
        self.register_buffer('v_max', torch.tensor(getattr(config, 'V_MAX', 1.1), dtype=torch.float32))
        
        # Base MVA for unit conversion (from normalizer, which gets it from config.yaml)
        self.base_mva = normalizer.base_mva

    def forward(self, 
                outputs_norm: torch.Tensor,
                targets_norm: torch.Tensor,
                measurements_norm: torch.Tensor,  # Not used in Denoising (we compare preds vs clean targets)
                ybus_batch: torch.Tensor,
                bus_types: torch.Tensor = None,  # Reserved for future use
                return_components: bool = False,
                epoch: int = None) -> torch.Tensor:  # Reserved for future use
        """
        Forward pass for loss calculation.
        
        Args:
            outputs_norm: Predicted clean state [batch, buses, 10] (Normalized)
            targets_norm: True clean state [batch, buses, 10] (Normalized)
            ybus_batch: Ybus matrices [batch, buses, buses] (p.u.)
        """
        # 1. Data Loss (L1) - MSE on Normalized Data
        # Normalized space ensures equal weighting across all 10 features
        # (voltages, powers, angles all contribute equally)
        l1_loss = F.mse_loss(outputs_norm, targets_norm)
        
        if not self.is_physics_informed:
            if return_components:
                return {
                    'total_loss': l1_loss,
                    'mse': l1_loss.item(),
                    'physics_loss': 0.0,
                    'safety_loss': 0.0
                }
            return l1_loss

        # 2. Physics Loss (L2) - Power Balance
        # Denormalize for physics calculation (need actual p.u. voltages for Ybus)
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
        
        # 4. Combine with Kendall's Homoscedastic Uncertainty Weighting
        # Loss = (w_i * L_i + log_var_i) where w_i = exp(-log_var_i)
        # L1 (MSE) is in normalized space, L2/L3 in physics space (p.u.)
        # Kendall's weights automatically learn to balance these different scales
        
        # Calculate precision weights (higher weight = more confident in that task)
        w_data = torch.exp(-self.log_vars[0])
        w_phys = torch.exp(-self.log_vars[1])
        w_safe = torch.exp(-self.log_vars[2])
        
        # Kendall's weighted loss formula (all components in physical units)
        loss = (w_data * l1_loss + self.log_vars[0]) + \
               (w_phys * l2_loss + self.log_vars[1]) + \
               (w_safe * l3_loss + self.log_vars[2])
               
        if return_components:
            return {
                'total_loss': loss,  # Keep as tensor for backprop!
                'mse': l1_loss.item(),
                'physics_loss': l2_loss.item(),
                'safety_loss': l3_loss.item(),
                'weights': [w_data.item(), w_phys.item(), w_safe.item()]
            }
            
        return loss
