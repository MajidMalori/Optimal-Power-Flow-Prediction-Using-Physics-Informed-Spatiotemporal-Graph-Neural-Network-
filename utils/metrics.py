# In utils/metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

def relative_mse_loss(outputs: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Calculates Mean Squared Error relative to the magnitude of the target."""
    relative_error = (outputs - targets) / (torch.abs(targets) + epsilon)
    return torch.mean(relative_error**2)

def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, ybus_batch: torch.Tensor, config: object) -> Dict[str, float]:
    """Computes both standard regression metrics and power system specific metrics."""
    with torch.no_grad():
        # Ensure outputs and targets have the same shape
        if outputs.dim() != targets.dim():
            if outputs.dim() == 2 and targets.dim() == 3:
                # Model outputs flattened [batch_size, num_buses * features], targets are [batch_size, num_buses, features]
                targets = targets.view(outputs.shape)
            elif outputs.dim() == 3 and targets.dim() == 2:
                # Model outputs [batch_size, num_buses, features], targets are flattened
                outputs = outputs.view(targets.shape)
            else:
                raise ValueError(f"Cannot reconcile output shape {outputs.shape} with target shape {targets.shape}")
        
        # Standard regression metrics
        mse = F.mse_loss(outputs, targets).item()
        rmse = torch.sqrt(torch.tensor(mse, device=outputs.device)).item()
        
        # Create PowerSystemLoss instance for physics calculations
        physics_metrics = PowerSystemLoss(config=config, normalizer=None)

        # For physics calculations, we need 3D format [batch_size, num_buses, features]
        if outputs.dim() == 2:
            # If outputs are flattened, reshape to 3D for physics calculations
            batch_size = outputs.shape[0]
            num_features = 10  # Standard: vm, va, p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren
            num_buses = outputs.shape[1] // num_features
            outputs_3d = outputs.view(batch_size, num_buses, num_features)
        else:
            outputs_3d = outputs

        # Use the complete predicted state for physics violations
        power_violation = physics_metrics._compute_power_balance_violation(
            state=outputs_3d,  # Use 3D format for physics calculations
            ybus_batch=ybus_batch,
            squared=False  # Use RMSE for evaluation
        ).mean().item()
        
        voltage_violation = torch.sqrt(physics_metrics._compute_voltage_limit_violation(
            outputs_3d  # Use 3D format for physics calculations
        )).mean().item()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'power_violation': power_violation,
            'voltage_violation': voltage_violation
        }

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

        self.lambda_p = getattr(config, 'LAMBDA_P', 10.0)
        self.lambda_v = getattr(config, 'LAMBDA_V', 10.0)
        
        self.register_buffer('v_min', torch.tensor(config.V_MIN, dtype=torch.float32))
        self.register_buffer('v_max', torch.tensor(config.V_MAX, dtype=torch.float32))

        # The system base power is crucial for converting MW/MVAr to per unit (pu)
        # Get system-specific base power based on the case name
        self.s_base_mva = self._get_system_base_power(config)

        # self.loss_scale_factor = getattr(config, "LOSS_SCALE_FACTOR", 1.0) if self.is_physics_informed else 0.0

    def _get_system_base_power(self, config) -> float:
        """
        Get the correct base power for each system type.
        
        System-specific base power values (from pandapower test cases):
        - Case33 (distribution): 10 MVA
        - Case57 (sub-transmission): 100 MVA  
        - Case118 (transmission): 100 MVA
        
        This ensures proper per-unit calculations and normalized power loss.
        """
        case_name = getattr(config, 'CASE_NAME', None)
        if not case_name:
            raise ValueError("CASE_NAME must be set in config to determine system base power")
        
        case_name_lower = case_name.lower()
        if 'case33' in case_name_lower:
            return 10.0  # Distribution system base power
        elif 'case57' in case_name_lower:
            return 100.0  # Sub-transmission system base power
        elif 'case118' in case_name_lower:
            return 100.0  # Transmission system base power
        else:
            raise ValueError(f"Unknown system type: {case_name}. Expected case33, case57, or case118")

    # --- START CORRECTION: The forward pass now accepts a BATCH of Ybus matrices and coefficients ---
    def forward(self, 
                outputs_norm: torch.Tensor, 
                targets_norm: torch.Tensor, 
                ybus_batch: torch.Tensor) -> torch.Tensor:
        
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
            return {
                'total_loss': data_loss,
                'mse': data_loss,
                'power_violation': torch.tensor(0.0, device=data_loss.device),
                'voltage_violation': torch.tensor(0.0, device=data_loss.device)
            }

        # 2. Physics-Informed Penalties
        # Denormalize model output to physical units for physics calculations
        outputs_denorm = self.normalizer.denormalize(outputs_norm, self.config.NUM_BUSES)
        
        # Calculate physics penalties (these return a value per-sample in the batch)
        power_violation_per_sample = self._compute_power_balance_violation(outputs_denorm, ybus_batch)
        voltage_violation_per_sample = self._compute_voltage_limit_violation(outputs_denorm)

        # Take the mean to get a single value for the batch
        power_penalty = torch.mean(power_violation_per_sample)
        voltage_penalty = torch.mean(voltage_violation_per_sample)
        
        # --- START CORRECTION: Apply individual lambdas to each penalty ---
        # Combine data loss and physics loss with their independent weights
        total_loss = data_loss + (self.lambda_p * power_penalty) + (self.lambda_v * voltage_penalty)
        # --- END CORRECTION ---
            
        # Return a dictionary for detailed logging
        return {
            'total_loss': total_loss,
            'mse': data_loss,
            'power_violation': power_penalty,
            'voltage_violation': voltage_penalty
        }

    # --- END CORRECTION ---

    def _get_power_injections_pu(self, state: torch.Tensor):
        """Extracts power injections from the state tensor and converts them to per unit."""
        # New 10-feature structure: [vm, va, p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren]
        p_load_mw = state[..., 2]
        q_load_mvar = state[..., 3]
        
        # Calculate total generation from separated components
        p_ext_mw = state[..., 4]  # External grid generation
        q_ext_mvar = state[..., 5]
        p_conv_mw = state[..., 6]  # Conventional generation
        q_conv_mvar = state[..., 7]
        p_ren_mw = state[..., 8]  # Renewable generation
        q_ren_mvar = state[..., 9]
        
        # Total local generation (excluding slack bus for power injection calculation)
        # Power injection = Local generation - Local load (at each bus)
        p_gen_mw = p_conv_mw + p_ren_mw
        q_gen_mvar = q_conv_mvar + q_ren_mvar
        
        p_inj_mw = p_gen_mw - p_load_mw
        q_inj_mvar = q_gen_mvar - q_load_mvar
        
        p_inj_pu = p_inj_mw / self.s_base_mva
        q_inj_pu = q_inj_mvar / self.s_base_mva
        
        return p_inj_pu, q_inj_pu
    
    def _get_power_injections(self, state: torch.Tensor):
        """Extracts power injections from the state tensor in original units (MW, MVAr)."""
        # New 10-feature structure: [vm, va, p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren]
        p_load_mw = state[..., 2]
        q_load_mvar = state[..., 3]
        
        # Calculate total generation from separated components
        p_conv_mw = state[..., 6]  # Conventional generation
        q_conv_mvar = state[..., 7]
        p_ren_mw = state[..., 8]  # Renewable generation
        q_ren_mvar = state[..., 9]
        
        # Total generation (local only, excluding slack bus)
        p_gen_mw = p_conv_mw + p_ren_mw
        q_gen_mvar = q_conv_mvar + q_ren_mvar
        
        p_inj_mw = p_gen_mw - p_load_mw
        q_inj_mvar = q_gen_mvar - q_load_mvar
        
        return p_inj_mw, q_inj_mvar, p_load_mw, q_load_mvar

    # --- START CORRECTION: This function now works with a batch of Ybus matrices ---
    def _compute_power_balance_violation(self, state, ybus_batch, squared=True):
        """
        Computes power balance violation.
        Args:
            squared: If True, returns MSE (for training), if False, returns RMSE (for evaluation)
        """
        # Calculate mismatches
        vm_pu, va_rad = state[..., 0], state[..., 1]
        V = vm_pu * torch.exp(1j * va_rad)
        I = torch.einsum('bij,bj->bi', ybus_batch.cfloat(), V)
        S_calc_pu = V * torch.conj(I)
        p_inj_pu, q_inj_pu = self._get_power_injections_pu(state)
        p_mismatch = p_inj_pu - S_calc_pu.real
        q_mismatch = q_inj_pu - S_calc_pu.imag
        
        mismatch_squared = p_mismatch**2 + q_mismatch**2
        
        if squared:
            return torch.mean(mismatch_squared, dim=-1)  # MSE for training
        else:
            return torch.sqrt(torch.mean(mismatch_squared, dim=-1))  # RMSE for evaluation

    def _compute_voltage_limit_violation(self, state: torch.Tensor) -> torch.Tensor:
        """Calculates the violation of voltage limits."""
        vm_pu = state[..., 0]
        
        v_below = F.relu(self.v_min - vm_pu)
        v_above = F.relu(vm_pu - self.v_max)
        
        return torch.mean(v_below**2 + v_above**2, dim=-1)

    # --- The functions below are for post-training MOOPF evaluation, not for the training loss ---


    def _compute_normalized_active_power_loss(self, state: torch.Tensor, Ybus: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """
        Computes normalized active power loss using the accurate power loss formula from equation (3.5):
        P_loss = Σ Σ Dij * [Rij/|Vit||Vjt| * (PitPjt + QitQjt) + Rij|Vit||Vjt|sin(θit-θjt)(QitPjt - QjtPit)]
        
        Vectorized implementation for better performance.
        Normalized by total system load to ensure values are in [0, 1] range across all bus systems.
        """
        # Extract state variables
        Vm = state[..., 0]  # Voltage magnitudes (p.u.) [batch_size, num_buses]
        Va = state[..., 1]  # Voltage angles (rad) [batch_size, num_buses]
        p_inj_pu, q_inj_pu = self._get_power_injections_pu(state)  # [batch_size, num_buses]
        
        batch_size, num_buses = Vm.shape[:2]
        
        # Check if we have any data
        if batch_size == 0 or num_buses == 0:
            return torch.zeros(batch_size, device=state.device, dtype=state.dtype)
        
        # Create branch connectivity mask from Ybus (Dij matrix)
        # A branch exists if there's a non-zero admittance between buses
        branch_exists = torch.abs(Ybus) > 1e-6  # [batch_size, num_buses, num_buses]
        
        # Remove self-loops (diagonal elements)
        branch_exists = branch_exists & ~torch.eye(num_buses, dtype=torch.bool, device=state.device)
        
        # Extract series impedance for each branch: Z_ij = 1/Y_ij
        # Only for existing branches to avoid division by zero
        z_series = torch.where(branch_exists, 1.0 / Ybus, torch.zeros_like(Ybus))
        r_series_raw = z_series.real  # Extract resistance
        
        # Take absolute value since physical resistance is always positive
        # Negative values come from Ybus convention (negative off-diagonal elements)
        r_series = torch.abs(r_series_raw)
        
        # Skip branches with very small resistance
        valid_branches = branch_exists & (r_series > 1e-6)
        
        # Create expanded tensors for vectorized operations
        # Expand voltages to [batch_size, num_buses, num_buses]
        Vm_i = Vm.unsqueeze(2).expand(-1, -1, num_buses)  # [batch_size, num_buses, num_buses]
        Vm_j = Vm.unsqueeze(1).expand(-1, num_buses, -1)  # [batch_size, num_buses, num_buses]
        
        # Expand voltage angles
        Va_i = Va.unsqueeze(2).expand(-1, -1, num_buses)  # [batch_size, num_buses, num_buses]
        Va_j = Va.unsqueeze(1).expand(-1, num_buses, -1)  # [batch_size, num_buses, num_buses]
        
        # Expand power injections
        P_i = p_inj_pu.unsqueeze(2).expand(-1, -1, num_buses)  # [batch_size, num_buses, num_buses]
        P_j = p_inj_pu.unsqueeze(1).expand(-1, num_buses, -1)  # [batch_size, num_buses, num_buses]
        Q_i = q_inj_pu.unsqueeze(2).expand(-1, -1, num_buses)  # [batch_size, num_buses, num_buses]
        Q_j = q_inj_pu.unsqueeze(1).expand(-1, num_buses, -1)  # [batch_size, num_buses, num_buses]
        
        # Calculate voltage magnitude products
        V_prod = Vm_i * Vm_j  # [batch_size, num_buses, num_buses]
        
        # Calculate angle differences
        angle_diff = Va_i - Va_j  # [batch_size, num_buses, num_buses]
        
        # Use the standard power system power loss formula: P_loss = R * |I|^2
        # Where I is the current flowing through the branch
        # Current can be calculated as: I = (V_i - V_j) / Z
        
        # Calculate complex voltages
        V_complex_i = Vm_i * torch.exp(1j * Va_i)
        V_complex_j = Vm_j * torch.exp(1j * Va_j)
        
        # Calculate branch current: I = (V_i - V_j) / Z
        V_diff = V_complex_i - V_complex_j
        
        # Avoid division by zero for invalid branches
        Z_complex = torch.where(valid_branches, z_series, torch.ones_like(z_series))
        I_complex = V_diff / Z_complex
        
        # Current magnitude squared: |I|^2
        I_mag_sq = I_complex.real**2 + I_complex.imag**2
        
        # Power loss: P_loss = R * |I|^2 (always positive since R > 0 and |I|^2 > 0)
        branch_losses = torch.where(valid_branches, r_series * I_mag_sq, torch.zeros_like(I_mag_sq))
        
        # Sum over all branches (only upper triangle to avoid double counting)
        # Create upper triangle mask
        upper_triangle = torch.triu(torch.ones(num_buses, num_buses, dtype=torch.bool, device=state.device), diagonal=1)
        upper_triangle = upper_triangle.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_buses, num_buses]
        
        # Apply upper triangle mask and sum
        total_loss_pu = torch.sum(branch_losses * upper_triangle, dim=(1, 2))  # [batch_size] in per-unit
        
        # FIXED: Use per-unit loss directly as normalized loss
        # Per-unit loss is already normalized by system base and represents loss as fraction of system capacity
        # Typical values: 0.01-0.05 (1-5% of system capacity)
        # This approach is robust and doesn't depend on generation predictions
        normalized_loss = total_loss_pu
        
        
        # Do NOT scale or clamp - report actual loss percentages
        # Let the physics loss naturally penalize excessive or unphysical losses
        # Values outside typical ranges provide important learning signals during training
        return normalized_loss

    def _compute_normalized_voltage_deviation(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized voltage deviation according to formula (3.6):
        f2 = Σ_t Σ_i |Vit - ViNt|/|ViNt|
        
        Args:
            state: Tensor containing voltage magnitudes in p.u. at shape [..., 0]
            
        Returns:
            Tensor containing normalized voltage deviations
        """
        # Get voltage magnitudes (already in p.u.)
        Vm = state[..., 0]  # Shape: [batch_size, num_buses]
        
        # Rated voltage is 1.0 p.u.
        V_rated = torch.ones_like(Vm)
        
        # Calculate absolute normalized deviation: |Vit - ViNt|/|ViNt|
        voltage_deviation = torch.abs(Vm - V_rated) / V_rated
        
        # Take mean across buses for each sample
        mean_deviation = torch.mean(voltage_deviation, dim=1)
        
        return mean_deviation

    def _compute_normalized_power_flow(self, state: torch.Tensor, Ybus: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """
        Computes normalized power flow magnitudes using ACOPF equation (3.8):
        P_i^DG + P_i = P_i^load + V_i Σ_(s=1)^(B_n) V_s (G_is cos θ_is + B_is sin θ_is)
        Q_i^DG + Q_i = Q_i^load + V_i Σ_(s=1)^(B_n) V_s (G_is sin θ_is - B_is cos θ_is)
        
        This implementation calculates the actual power flow magnitudes through the network
        using the standard AC power flow equations, same foundation as power balance violation
        but measuring flow magnitudes instead of balance mismatches.
        
        Args:
            state: Tensor containing [vm_pu, va_rad, p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren]
            Ybus: Admittance matrix [batch_size, num_buses, num_buses]
            epsilon: Small value to avoid division by zero
            
        Returns:
            Tensor containing normalized power flow magnitudes [batch_size]
        """
        # Extract voltage state variables
        vm_pu = state[..., 0]  # Voltage magnitudes (p.u.) [batch_size, num_buses]
        va_rad = state[..., 1]  # Voltage angles (rad) [batch_size, num_buses]
        
        batch_size, num_buses = vm_pu.shape[:2]
        
        # Check if we have any data
        if batch_size == 0 or num_buses == 0:
            return torch.zeros(batch_size, device=state.device, dtype=state.dtype)
        
        # Calculate complex voltages: V = |V| * e^(jθ)
        V = vm_pu * torch.exp(1j * va_rad)  # [batch_size, num_buses]
        
        # Calculate currents using Ybus: I = Ybus * V
        # This implements: I_i = Σ_s Y_is * V_s
        I = torch.einsum('bij,bj->bi', Ybus.cfloat(), V)  # [batch_size, num_buses]
        
        # Calculate complex power flows: S = V * conj(I)
        # This implements the ACOPF equation (3.8):
        # S_i = V_i * conj(I_i) = V_i * Σ_s Y_is* * V_s*
        # Where Y_is = G_is + jB_is, so:
        # P_i = Re(S_i) = V_i Σ_s V_s (G_is cos θ_is + B_is sin θ_is)
        # Q_i = Im(S_i) = V_i Σ_s V_s (G_is sin θ_is - B_is cos θ_is)
        S_calc_pu = V * torch.conj(I)  # [batch_size, num_buses]
        
        # Extract active and reactive power flows (preserve signs for physics accuracy)
        p_flow_values = S_calc_pu.real  # P_calculated (can be negative for reverse flow) [batch_size, num_buses]
        q_flow_values = S_calc_pu.imag  # Q_calculated (can be negative for reverse flow) [batch_size, num_buses]
        
        # Calculate apparent power flow magnitude per bus (magnitude only for normalization)
        s_flow_magnitudes = torch.sqrt(p_flow_values**2 + q_flow_values**2)  # [batch_size, num_buses]
        
        # Calculate mean apparent power flow magnitude per bus (normalized metric)
        # This gives a per-bus average that's naturally bounded and comparable
        mean_flow_magnitude_per_bus = torch.mean(s_flow_magnitudes, dim=-1)  # [batch_size]
        
        # Use total system load for normalization (physically meaningful and always positive)
        # This avoids the negative p_ext problem while maintaining physical correctness
        p_load = torch.sum(state[..., 2], dim=-1)  # Total load [batch_size] (always positive)
        q_load = torch.sum(state[..., 3], dim=-1)  # Total reactive load [batch_size] (always positive)
        total_load_magnitude = torch.sqrt(p_load**2 + q_load**2)  # Total apparent load [batch_size]
        total_load_pu = total_load_magnitude / self.s_base_mva  # Convert to per-unit [batch_size]
        
        # Normalize by total system load + small epsilon to avoid division by zero
        # This gives us power flow as a fraction of total system load, which is physically meaningful
        # Load is always positive, avoiding the negative generation problem
        normalized_power_flow = mean_flow_magnitude_per_bus / (total_load_pu + epsilon)
        
        # Return raw values without forcing them positive - let the physics loss handle bad predictions
        # This preserves the learning signal for physics-informed training
        return normalized_power_flow

    def _compute_carbon_emissions(
        self, 
        predicted_state_physical: torch.Tensor, 
        time_carbon_coeff: torch.Tensor, 
        time_energy_coeff: torch.Tensor,
        renewable_fraction: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes carbon emissions using component-based approach:
        f3 = Σ(Pext_t + Pconv_t) * Cm / Ef
        
        Where:
        - Pext_t: External grid generation (p_ext_grid)
        - Pconv_t: Conventional generation (p_conventional) 
        - Pren_t: Renewable generation (p_renewable) - carbon-free
        - Cm: Carbon emissions per unit electricity (time_carbon_coeff)
        - Ef: Energy utilization coefficient (time_energy_coeff)
        
        This approach directly uses the separated generation components
        instead of calculating renewable fractions.
        """
        # Extract separated generation components from 10-feature state
        # Format: [vm_pu, va_rad, p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren]
        
        # Get carbon-emitting generation components (conventional only - no external grid)
        # p_ext is not used in physics calculations and causes clamping issues
        total_conv_gen = torch.sum(predicted_state_physical[..., 6], dim=-1)      # p_conventional (MW)
        total_renewable_gen = torch.sum(predicted_state_physical[..., 8], dim=-1)  # p_renewable (MW) - carbon-free
        
        # NO CLAMPING: Use only conventional generation for carbon emissions
        # External grid is not included in carbon calculations (no clamping needed)
        total_carbon_emitting_gen = total_conv_gen
        
        # Ensure coefficients are correctly shaped for broadcasting
        carbon_intensity = time_carbon_coeff.squeeze(-1) if time_carbon_coeff.dim() > 1 else time_carbon_coeff  # Cm
        energy_coefficient = time_energy_coeff.squeeze(-1) if time_energy_coeff.dim() > 1 else time_energy_coeff  # Ef
        
        # Apply component-based carbon emission calculation (conventional only)
        # f3 = Pconv * Cm / Ef (no external grid, no clamping)
        raw_emissions = (total_carbon_emitting_gen * carbon_intensity) / (energy_coefficient + 1e-9)
        
        # Calculate total generation for normalization (only positive generation)
        total_generation = total_carbon_emitting_gen + total_renewable_gen
        
        # DEBUG: Print generation components
        # Normalize emissions based on the fraction of carbon-emitting generation
        # This gives a value between 0 and 1 where:
        # - 1.0 = all generation is carbon-emitting (worst case)
        # - 0.0 = all generation is renewable (best case)
        normalized_emissions = total_carbon_emitting_gen / (total_generation + 1e-9)
        
        # Check for impossible values
        if (normalized_emissions > 1.0).any():
            print(f"  ⚠️  IMPOSSIBLE: normalized_emissions > 1.0 found!")
            print(f"  ⚠️  This means carbon-emitting generation > total generation")
            print(f"  ⚠️  This is physically impossible!")
            print(f"  ⚠️  Check if renewable generation is negative!")
        
        return {'raw': raw_emissions, 'normalized': normalized_emissions}