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
        self.log_vars = nn.Parameter(torch.zeros(4))  # [Data, Physics, Safety, Constraint]
        
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
        
        # Extract State Variables (Physical Units) - use indexing to avoid copies
        # Voltage Magnitude (p.u.)
        vm_pu = preds_phys[..., FeatureIndices.VM]
        # Voltage Angle (radians)
        va_rad = preds_phys[..., FeatureIndices.VA]
        
        # Construct Complex Voltage (in-place where possible)
        V = vm_pu * torch.exp(1j * va_rad)
        
        # Calculate Theoretical Power Flow (Injection)
        # S = V * conj(Y * V)
        # Ybus is in p.u.
        ybus_complex = ybus_batch if ybus_batch.is_complex() else ybus_batch.to(torch.complex64)
        
        # Optimized: Use matmul instead of einsum for better memory efficiency
        # I_inj = Y @ V
        I_inj = torch.bmm(ybus_complex, V.unsqueeze(-1)).squeeze(-1)
        S_flow_pu = V * torch.conj(I_inj)
        P_flow_pu = S_flow_pu.real
        Q_flow_pu = S_flow_pu.imag
        
        # Extract Predicted Net Injections (Physical Units -> Convert to p.u.)
        # Use direct indexing to avoid intermediate copies
        p_load = preds_phys[..., FeatureIndices.P_LOAD]
        p_ext = preds_phys[..., FeatureIndices.P_EXT_GRID]
        p_conv = preds_phys[..., FeatureIndices.P_CONV]
        p_ren = preds_phys[..., FeatureIndices.P_REN]
        
        q_load = preds_phys[..., FeatureIndices.Q_LOAD]
        q_ext = preds_phys[..., FeatureIndices.Q_EXT_GRID]
        q_conv = preds_phys[..., FeatureIndices.Q_CONV]
        q_ren = preds_phys[..., FeatureIndices.Q_REN]
        
        # Calculate Net Injection (MW/MVar) - use in-place operations where possible
        P_net_mw = (p_ext + p_conv + p_ren) - p_load
        Q_net_mvar = (q_ext + q_conv + q_ren) - q_load
        
        # Convert to p.u. (in-place division)
        P_net_pu = P_net_mw / self.base_mva
        Q_net_pu = Q_net_mvar / self.base_mva
        
        # L2 Loss: MSE of Mismatch (compute directly without storing intermediate)
        l2_loss = F.mse_loss(P_net_pu, P_flow_pu, reduction='mean') + F.mse_loss(Q_net_pu, Q_flow_pu, reduction='mean')
        
        # 3. Safety Loss (L3) - Voltage Limits
        # Soft constraints on Voltage Magnitude
        # ReLU(|V| - 1.1)^2 + ReLU(0.9 - |V|)^2
        v_upper_violation = F.relu(vm_pu - self.v_max)
        v_lower_violation = F.relu(self.v_min - vm_pu)
        l3_loss = torch.mean(v_upper_violation**2 + v_lower_violation**2)
        
        # 4. Physics Constraint Loss (L4) - Non-negative Physical Quantities
        # P_conv, P_ren, and VM must be non-negative (physical constraints)
        # FIX: Calculate on NORMALIZED values to prevent loss explosion due to unit scaling (MW vs p.u.)
        # Normalized P_conv and P_ren should be >= 0 (since they are scaled by 1/BaseMVA)
        
        # Extract normalized quantities directly
        p_conv_norm = outputs_norm[..., FeatureIndices.P_CONV]
        p_ren_norm = outputs_norm[..., FeatureIndices.P_REN]
        
        # Penalize negative normalized power (equivalent to negative physical power but scaled)
        p_conv_violation = F.relu(-p_conv_norm)
        p_ren_violation = F.relu(-p_ren_norm)
        
        # For VM, we still check physical value because normalized VM can be negative (centered at 1.0)
        # VM_phys = (VM_norm / 10) + 1.0. Constraint: VM_phys >= 0.
        # But VM explosion is rare compared to Power. We can keep physical check for VM or use normalized threshold.
        # Using physical for VM is fine as it's ~1.0.
        vm_negative_violation = F.relu(-vm_pu) 
        
        l4_loss = torch.mean(p_conv_violation**2 + p_ren_violation**2 + vm_negative_violation**2)
        
        # 5. Combine with Kendall's Homoscedastic Uncertainty Weighting
        # Loss = (w_i * L_i + log_var_i) where w_i = exp(-log_var_i)
        # L1 (MSE) is in normalized space, L2/L3/L4 in physics space (p.u.)
        # Kendall's weights automatically learn to balance these different scales
        #
        # NOTE: The total loss CAN be negative because log_var_i can be negative.
        # This is mathematically correct in Kendall's method - the loss is actually
        # a log-likelihood that can be negative. The optimizer will still work correctly.
        # We do NOT clamp log_vars - that would be "cheating" and break the optimization.
        
        # Calculate precision weights (higher weight = more confident in that task)
        w_data = torch.exp(-self.log_vars[0])
        w_phys = torch.exp(-self.log_vars[1])
        w_safe = torch.exp(-self.log_vars[2])
        w_constraint = torch.exp(-self.log_vars[3])
        
        # Kendall's weighted loss formula (all components in physical units)
        # Total loss can be negative if log_vars are very negative - this is mathematically correct
        loss = (w_data * l1_loss + self.log_vars[0]) + \
               (w_phys * l2_loss + self.log_vars[1]) + \
               (w_safe * l3_loss + self.log_vars[2]) + \
               (w_constraint * l4_loss + self.log_vars[3])
               
        if return_components:
            return {
                'total_loss': loss,  # Keep as tensor for backprop!
                'mse': l1_loss.item(),
                'physics_loss': l2_loss.item(),
                'safety_loss': l3_loss.item(),
                'constraint_loss': l4_loss.item(),
                'weights': [w_data.item(), w_phys.item(), w_safe.item(), w_constraint.item()]
            }
            
        return loss

def compute_moopf_metrics(preds_phys: torch.Tensor, ybus_batch: torch.Tensor, base_mva: float) -> dict:
    """
    Compute Multi-Objective Optimal Power Flow (MOOPF) Metrics.
    
    Metric A: Carbon Intensity Score (0.0 - 1.0)
    Metric B: Power Loss Score (Normalized)
    Metric C: Voltage Stability Score (p.u.)
    
    Args:
        preds_phys: Denormalized predictions [batch, buses, 10]
        ybus_batch: Ybus matrices [batch, buses, buses] (p.u.)
        base_mva: System base power (MVA)
        
    Returns:
        Dictionary of metrics
    """
    # Extract features
    p_load = preds_phys[..., FeatureIndices.P_LOAD]
    p_ext = preds_phys[..., FeatureIndices.P_EXT_GRID]
    p_conv = preds_phys[..., FeatureIndices.P_CONV]
    p_ren = preds_phys[..., FeatureIndices.P_REN]
    vm_pu = preds_phys[..., FeatureIndices.VM]
    va_rad = preds_phys[..., FeatureIndices.VA]
    
    # --- Metric A: Carbon Intensity Score ---
    # Fossil = Sum(P_conv) + Sum(ReLU(P_ext))
    # Total = Fossil + Sum(P_ren)
    # Score = Fossil / Total
    
    # Sum over buses (vectorized: combine all sums in one operation)
    total_p_conv = p_conv.sum(dim=1)
    total_p_ext = F.relu(p_ext).sum(dim=1)  # Only imports count as generation
    total_p_ren = p_ren.sum(dim=1)
    
    fossil_gen = total_p_conv + total_p_ext
    total_gen = fossil_gen + total_p_ren
    
    carbon_score = fossil_gen / (total_gen + 1e-6)
    
    # --- Metric B: Power Loss Score ---
    # Loss_MW = P_gen - P_load (from actual power values)
    # Load_MW = Sum(P_load)
    # Score = Loss_MW / Load_MW
    # 
    # Note: We calculate losses from actual power values (P_gen - P_load),
    # not from voltages (sum(Real(V*conj(Y*V)))), because predicted voltages
    # may not satisfy power balance during training. For a balanced system,
    # both methods give the same result, but P_gen - P_load is more reliable.
    
    # Calculate losses from actual power values (always positive for passive network)
    total_gen_mw = (p_ext + p_conv + p_ren).sum(dim=1)  # Sum all generation
    total_load_mw = p_load.sum(dim=1)  # Sum all loads
    total_losses_mw = total_gen_mw - total_load_mw  # Losses = Gen - Load (must be >= 0)
    
    power_loss_score = total_losses_mw / (total_load_mw + 1e-6)
    
    # --- Metric C: Voltage Stability Score ---
    # Score = Mean(|V_mag - 1.0|) per sample
    voltage_stability = torch.abs(vm_pu - 1.0).mean(dim=1)
    
    # Return per-sample metrics (not batch averages)
    return {
        'carbon_score': carbon_score,  # [batch_size] tensor
        'power_loss_score': power_loss_score,  # [batch_size] tensor
        'voltage_stability_score': voltage_stability  # [batch_size] tensor
    }
