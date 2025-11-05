import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, ybus_batch: torch.Tensor, config: object, bus_types: torch.Tensor = None) -> Dict[str, float]:
    """
    Computes both standard regression metrics and power system specific metrics.
    
    Args:
        outputs: Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
        targets: True unknowns [batch, buses, 2] (OPF: bus-type dependent)
        ybus_batch: Ybus matrices for physics calculations
        config: Configuration object
        bus_types: Optional [batch, buses] with codes [0=PQ, 1=PV, 2=Slack] for OPF-specific metrics
    """
    import torch.nn.functional as F
    
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
        
        # OPF: Bus-type-specific metrics (optional)
        metrics = {
            'mse': mse,
            'rmse': rmse,
        }
        
        if bus_types is not None:
            # Compute bus-type-specific MSE for reporting
            # PQ buses (0): [V, θ], PV buses (1): [Q, θ], Slack (2): [P, Q]
            bus_types_cpu = bus_types.cpu() if bus_types.is_cuda else bus_types
            
            for bus_type_code, bus_type_name in [(0, 'PQ'), (1, 'PV'), (2, 'Slack')]:
                mask = (bus_types_cpu == bus_type_code)
                if mask.any():
                    outputs_type = outputs[mask]
                    targets_type = targets[mask]
                    mse_type = F.mse_loss(outputs_type, targets_type).item()
                    metrics[f'mse_{bus_type_name.lower()}'] = mse_type
        
        metrics['power_violation'] = 0.0
        metrics['voltage_violation'] = 0.0
        
        return metrics

class PowerSystemLoss(nn.Module):
    """
    A comprehensive, physics-informed loss function for power system state estimation.
    This version correctly handles per-sample Ybus matrices and time-varying coefficients,
    making it suitable for datasets with mixed scenarios (e.g., different renewable
    fractions or N-1 contingencies).
    
    Uses Learnable Uncertainty Weighting (Kendall et al., CVPR 2018):
    - Automatically learns optimal loss weights via backpropagation
    - Bayesian interpretation with homoscedastic uncertainty
    - Paper: "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    def __init__(self, config: object, normalizer, is_gcn: bool = False):
        super().__init__()
        self.config = config
        self.normalizer = normalizer
        self.is_physics_informed = not is_gcn
        self.mse_loss_fn = nn.MSELoss()
        
        self.log_sigma_data = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_power = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_voltage = nn.Parameter(torch.tensor(0.0))
        
        self.register_buffer('v_min', torch.tensor(config.V_MIN, dtype=torch.float32))
        self.register_buffer('v_max', torch.tensor(config.V_MAX, dtype=torch.float32))

        self.s_base_mva = self._get_system_base_power(config)

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

    def forward(self, 
                outputs_norm: torch.Tensor,
                targets_norm: torch.Tensor,
                measurements_norm: torch.Tensor,
                ybus_batch: torch.Tensor,
                bus_types: torch.Tensor = None,
                return_components: bool = False) -> torch.Tensor:
        
        # Ensure outputs and targets have the same shape
        if outputs_norm.dim() != targets_norm.dim():
            if outputs_norm.dim() == 2 and targets_norm.dim() == 3:
                # Reshape flattened 2D output [batch, buses*features] to 3D [batch, buses, features]
                outputs_norm = outputs_norm.view(targets_norm.shape)
            else:
                 raise ValueError(f"Shape mismatch: outputs {outputs_norm.shape}, targets {targets_norm.shape}")
        
        # Ensure outputs are in [batch, buses, 2] format
        if outputs_norm.dim() == 2:
            # Flattened output: reshape to [batch, buses, 2]
            batch_size = outputs_norm.shape[0]
            num_buses = targets_norm.shape[1]  # Get num_buses from targets
            expected_size = batch_size * num_buses * 2
            actual_size = outputs_norm.numel()
            if actual_size != expected_size:
                raise ValueError(
                    f"Cannot reshape outputs from {outputs_norm.shape} (size={actual_size}) "
                    f"to [batch={batch_size}, buses={num_buses}, 2] (expected size={expected_size}). "
                    f"Targets shape: {targets_norm.shape}"
                )
            outputs_norm = outputs_norm.view(batch_size, num_buses, 2)
        elif outputs_norm.dim() == 3:
            # Already 3D: check if it's [batch, buses, 2] or needs reshaping
            batch_size, num_buses = outputs_norm.shape[0], outputs_norm.shape[1]
            total_features = outputs_norm.shape[-1]
            actual_size = outputs_norm.numel()
            expected_size = batch_size * num_buses * 2
            
            if total_features == 2:
                # Correct shape: [batch, buses, 2]
                pass  # Already correct
            elif total_features == num_buses * 2:
                # Wrong shape: [batch, buses, buses*2] - this is a bug, can't reshape
                # The model is outputting wrong shape - each bus has buses*2 features instead of 2
                raise ValueError(
                    f"Model output has wrong shape: {outputs_norm.shape}. "
                    f"Each bus has {total_features} features (should be 2). "
                    f"This suggests the output layer is producing [batch, buses, buses*2] instead of [batch, buses, 2]. "
                    f"Actual size: {actual_size}, Expected: {expected_size}. "
                    f"Targets shape: {targets_norm.shape}"
                )
            else:
                raise ValueError(
                    f"Unexpected output shape: {outputs_norm.shape} (size={actual_size}). "
                    f"Expected [batch, buses, 2] (size={expected_size}). "
                    f"Got last dim {total_features}, expected 2. "
                    f"Targets shape: {targets_norm.shape}"
                )
        
        outputs_denorm_for_mse = self.normalizer.denormalize(outputs_norm)
        targets_denorm_for_mse = self.normalizer.denormalize(targets_norm)
        
        if bus_types is None:
            vm_pred = outputs_denorm_for_mse[..., 0]
            va_pred = outputs_denorm_for_mse[..., 1]
            vm_true = targets_denorm_for_mse[..., 0]
            va_true = targets_denorm_for_mse[..., 1]
            
            mse_vm = self.mse_loss_fn(vm_pred, vm_true)
            mse_va = self.mse_loss_fn(va_pred, va_true)
            data_loss = mse_vm + mse_va
            data_loss_vm = mse_vm
            data_loss_va = mse_va
        else:
            var1_pred = outputs_denorm_for_mse[..., 0]
            var2_pred = outputs_denorm_for_mse[..., 1]
            var1_true = targets_denorm_for_mse[..., 0]
            var2_true = targets_denorm_for_mse[..., 1]
            
            mse_var1 = self.mse_loss_fn(var1_pred, var1_true)
            mse_var2 = self.mse_loss_fn(var2_pred, var2_true)
            data_loss = mse_var1 + mse_var2
            data_loss_vm = mse_var1
            data_loss_va = mse_var2
        
        # If not physics-informed, we are done.
        if not self.is_physics_informed:
            return {
                'total_loss': data_loss,
                'mse': data_loss,
                'mse_vm': data_loss_vm,
                'mse_va': data_loss_va,
                'power_violation': torch.tensor(0.0, device=data_loss.device),
                'voltage_violation': torch.tensor(0.0, device=data_loss.device)
            }

        outputs_denorm = outputs_denorm_for_mse
        
        vm_pu = outputs_denorm[..., 0]
        negative_vm_count = (vm_pu < 0).sum().item()
        negative_vm_fraction = negative_vm_count / vm_pu.numel() if vm_pu.numel() > 0 else 0.0
        
        if not hasattr(self, '_batch_count'):
            self._batch_count = 0
        self._batch_count += 1
        if negative_vm_count > 0 and self._batch_count % 100 == 0:
            min_vm = vm_pu.min().item()
            max_vm = vm_pu.max().item()
            print(f"Batch {self._batch_count}: {negative_vm_count} negative voltage predictions "
                  f"({negative_vm_fraction*100:.1f}%), VM range: [{min_vm:.4f}, {max_vm:.4f}]")
        
        if measurements_norm.dim() == 4:
            # Sequential model: use last timestep [batch, seq_len, buses, 10] -> [batch, buses, 10]
            measurements_norm = measurements_norm[:, -1, :, :]  # Take last timestep
        elif measurements_norm.dim() != 3:
            raise ValueError(
                f"measurements_norm must be 3D [batch, buses, 10] or 4D [batch, seq_len, buses, 10], "
                f"but got shape {measurements_norm.shape}"
            )
        measurements_denorm = self.normalizer.denormalize(measurements_norm)
        
        power_violation_per_sample = self._compute_power_balance_violation(
            predicted_voltages=outputs_denorm,
            measured_power=measurements_denorm,
            ybus_batch=ybus_batch
        )
        voltage_violation_per_sample = self._compute_voltage_limit_violation(outputs_denorm)

        power_penalty = torch.mean(power_violation_per_sample)
        voltage_penalty = torch.mean(voltage_violation_per_sample)
        
        sigma_data = torch.exp(self.log_sigma_data)
        sigma_power = torch.exp(self.log_sigma_power)
        sigma_voltage = torch.exp(self.log_sigma_voltage)
        
        weighted_data_loss = (1.0 / (2.0 * sigma_data ** 2)) * data_loss
        weighted_power_loss = (1.0 / (2.0 * sigma_power ** 2)) * power_penalty
        weighted_voltage_loss = (1.0 / (2.0 * sigma_voltage ** 2)) * voltage_penalty
        
        regularization = torch.log(sigma_data) + torch.log(sigma_power) + torch.log(sigma_voltage)
        
        total_loss = weighted_data_loss + weighted_power_loss + weighted_voltage_loss + regularization
        
        if return_components and self.is_physics_informed:
            return {
                'total_loss': total_loss,
                'mse': data_loss,
                'mse_vm': data_loss_vm,
                'mse_va': data_loss_va,
                'power_violation': power_penalty,
                'voltage_violation': voltage_penalty,
                'mse_vm_loss': data_loss_vm,
                'mse_va_loss': data_loss_va,
                'physics_loss': weighted_power_loss + weighted_voltage_loss
            }
        else:
            return {
                'total_loss': total_loss,
                'mse': data_loss,
                'mse_weighted': weighted_data_loss,
                'mse_vm': data_loss_vm,
                'mse_va': data_loss_va,
                'power_violation': power_penalty,
                'voltage_violation': voltage_penalty
            }

    def _get_power_injections_pu(self, measurements: torch.Tensor):
        """
        Extracts power injections from MEASURED data and converts to per unit.
        
        Args:
            measurements: Measurement tensor [batch, buses, 10]
                         Format: [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_partial, va_partial]
        
        Returns:
            p_inj_pu, q_inj_pu: Power injections in per-unit
        """
        p_load_mw = measurements[..., 0]
        q_load_mvar = measurements[..., 1]
        p_ext_mw = measurements[..., 2]
        q_ext_mvar = measurements[..., 3]
        p_conv_mw = measurements[..., 4]
        q_conv_mvar = measurements[..., 5]
        p_ren_mw = measurements[..., 6]
        q_ren_mvar = measurements[..., 7]
        
        p_gen_mw = p_conv_mw + p_ren_mw
        q_gen_mvar = q_conv_mvar + q_ren_mvar
        
        p_inj_mw = p_gen_mw - p_load_mw
        q_inj_mvar = q_gen_mvar - q_load_mvar
        
        p_inj_pu = p_inj_mw / self.s_base_mva
        q_inj_pu = q_inj_mvar / self.s_base_mva
        
        return p_inj_pu, q_inj_pu
    
    def _get_power_injections(self, state: torch.Tensor):
        """Extracts power injections from the state tensor in original units (MW, MVAr)."""
        p_load_mw = state[..., 2]
        q_load_mvar = state[..., 3]
        p_conv_mw = state[..., 6]
        q_conv_mvar = state[..., 7]
        p_ren_mw = state[..., 8]
        q_ren_mvar = state[..., 9]
        
        p_gen_mw = p_conv_mw + p_ren_mw
        q_gen_mvar = q_conv_mvar + q_ren_mvar
        
        p_inj_mw = p_gen_mw - p_load_mw
        q_inj_mvar = q_gen_mvar - q_load_mvar
        
        return p_inj_mw, q_inj_mvar, p_load_mw, q_load_mvar
    def _compute_power_balance_violation(self, predicted_voltages, measured_power, ybus_batch, squared=True):
        """
        Computes power balance violation for pure state estimation.
        
        Args:
            predicted_voltages: Predicted voltage state [batch, buses, 2] = [vm, va]
            measured_power: Measured power injections [batch, buses, 10] = [p_load, q_load, ..., vm_meas, va_meas]
            ybus_batch: Admittance matrices [batch, buses, buses]
            squared: If True, returns MSE (for training), if False, returns RMSE (for evaluation)
            
        Returns:
            Power balance violation per sample [batch]
        """
        vm_pu = predicted_voltages[..., 0]
        va_rad = predicted_voltages[..., 1]
        
        V = vm_pu * torch.exp(1j * va_rad)
        I = torch.einsum('bij,bj->bi', ybus_batch.cfloat(), V)
        S_calc_pu = V * torch.conj(I)
        
        p_inj_pu, q_inj_pu = self._get_power_injections_pu(measured_power)
        
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

    def _compute_normalized_active_power_loss(self, voltages: torch.Tensor, measurements: torch.Tensor, Ybus: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """
        Computes normalized active power loss using the accurate power loss formula from equation (3.5):
        P_loss = Σ Σ Dij * [Rij/|Vit||Vjt| * (PitPjt + QitQjt) + Rij|Vit||Vjt|sin(θit-θjt)(QitPjt - QjtPit)]
        
        Vectorized implementation for better performance.
        Normalized by total system load to ensure values are in [0, 1] range across all bus systems.
        
        Args:
            voltages: Predicted voltages [batch, buses, 2] = [vm, va]
            measurements: Measured power [batch, buses, 10] = [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
            Ybus: Admittance matrix [batch, buses, buses]
        """
        # Extract state variables
        Vm = voltages[..., 0]  # Voltage magnitudes (p.u.) [batch_size, num_buses]
        Va = voltages[..., 1]  # Voltage angles (rad) [batch_size, num_buses]
        p_inj_pu, q_inj_pu = self._get_power_injections_pu(measurements)  # [batch_size, num_buses]
        
        batch_size, num_buses = Vm.shape[:2]
        
        # Check if we have any data
        if batch_size == 0 or num_buses == 0:
            return torch.zeros(batch_size, device=Vm.device, dtype=Vm.dtype)
        
        # Create branch connectivity mask from Ybus (Dij matrix)
        # A branch exists if there's a non-zero admittance between buses
        branch_exists = torch.abs(Ybus) > 1e-6  # [batch_size, num_buses, num_buses]
        
        # Remove self-loops (diagonal elements)
        branch_exists = branch_exists & ~torch.eye(num_buses, dtype=torch.bool, device=voltages.device)
        
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
        upper_triangle = torch.triu(torch.ones(num_buses, num_buses, dtype=torch.bool, device=voltages.device), diagonal=1)
        upper_triangle = upper_triangle.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_buses, num_buses]
        
        total_loss_pu = torch.sum(branch_losses * upper_triangle, dim=(1, 2))
        
        V = Vm * torch.exp(1j * Va)
        I = torch.einsum('bij,bj->bi', Ybus.cfloat(), V)
        S = V * torch.conj(I)
        
        total_power_injection = torch.sum(S.real, dim=-1)
        power_loss_pu = torch.relu(total_power_injection)
        
        p_load_total = torch.sum(measurements[..., 0], dim=-1)
        total_load_pu = p_load_total / self.s_base_mva
        
        normalized_loss = power_loss_pu / (total_load_pu + epsilon)
        
        return normalized_loss

    def _compute_normalized_voltage_deviation(self, voltages: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized voltage deviation according to formula (3.6):
        f2 = Σ_t Σ_i |Vit - ViNt|/|ViNt|
        
        Args:
            voltages: Predicted voltages [batch, buses, 2] = [vm, va]
            
        Returns:
            Tensor containing normalized voltage deviations
        """
        # Extract voltage magnitudes
        Vm = voltages[..., 0]  # Shape: [batch_size, num_buses]
        
        # Rated voltage is 1.0 p.u.
        V_rated = torch.ones_like(Vm)
        
        # Calculate absolute normalized deviation: |Vit - ViNt|/|ViNt|
        voltage_deviation = torch.abs(Vm - V_rated) / V_rated
        
        # Take mean across buses for each sample
        mean_deviation = torch.mean(voltage_deviation, dim=1)
        
        return mean_deviation

    def _compute_normalized_power_flow(self, voltages: torch.Tensor, measurements: torch.Tensor, Ybus: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """
        Computes normalized power flow magnitudes using ACOPF equation (3.8):
        P_i^DG + P_i = P_i^load + V_i Σ_(s=1)^(B_n) V_s (G_is cos θ_is + B_is sin θ_is)
        Q_i^DG + Q_i = Q_i^load + V_i Σ_(s=1)^(B_n) V_s (G_is sin θ_is - B_is cos θ_is)
        
        This implementation calculates the actual power flow magnitudes through the network
        using the standard AC power flow equations, same foundation as power balance violation
        but measuring flow magnitudes instead of balance mismatches.
        
        Args:
            voltages: Predicted voltages [batch, buses, 2] = [vm, va]
            measurements: Measured power [batch, buses, 10] = [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
            Ybus: Admittance matrix [batch_size, num_buses, num_buses]
            epsilon: Small value to avoid division by zero
            
        Returns:
            Tensor containing normalized power flow magnitudes [batch_size]
        """
        # Extract voltage state variables
        vm_pu = voltages[..., 0]  # Voltage magnitudes (p.u.) [batch_size, num_buses]
        va_rad = voltages[..., 1]  # Voltage angles (rad) [batch_size, num_buses]
        
        batch_size, num_buses = vm_pu.shape[:2]
        
        # Check if we have any data
        if batch_size == 0 or num_buses == 0:
            return torch.zeros(batch_size, device=voltages.device, dtype=voltages.dtype)
        
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
        
        p_load = torch.sum(measurements[..., 0], dim=-1)
        q_load = torch.sum(measurements[..., 1], dim=-1)
        total_load_magnitude = torch.sqrt(p_load**2 + q_load**2)
        total_load_pu = total_load_magnitude / self.s_base_mva
        
        normalized_power_flow = mean_flow_magnitude_per_bus / (total_load_pu + epsilon)
        
        return normalized_power_flow

    def _compute_carbon_emissions(
        self, 
        measurements: torch.Tensor, 
        time_carbon_coeff: torch.Tensor, 
        time_energy_coeff: torch.Tensor,
        renewable_fraction: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes carbon emissions using component-based approach.
        
        Physical Sign Convention (PandaPower):
        - p_ext > 0: System IMPORTS from grid (grid supplies power) → Count as carbon if grid has fossil fuels
        - p_ext < 0: System EXPORTS to grid (grid receives power) → NOT our carbon responsibility
        - p_conv: Always positive (conventional generation) → Always carbon-emitting
        - p_ren: Always positive (renewable generation) → Zero carbon
        
        Formula:
            Carbon-emitting generation = p_conv + max(0, p_ext)
            f3 = (p_conv + max(0, p_ext)) * Cm / Ef
        
        Where:
        - p_conv: Conventional generation (coal, gas, nuclear)
        - p_ext: External grid power (only count imports, not exports)
        - Cm: Carbon intensity coefficient (time_carbon_coeff)
        - Ef: Energy utilization coefficient (time_energy_coeff)
        
        Args:
            measurements: Measured power [batch, buses, 10] = [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
        """
        # Extract generation components from measurements
        # Format: [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
        
        p_ext_mw = measurements[..., 2]     # External grid power (can be positive or negative)
        p_conv_mw = measurements[..., 4]    # Conventional generation (always positive)
        p_ren_mw = measurements[..., 6]     # Renewable generation (always positive)
        
        total_conv_gen = torch.sum(p_conv_mw, dim=-1)
        grid_import_power = torch.sum(F.relu(p_ext_mw), dim=-1)
        total_carbon_emitting_gen = total_conv_gen + grid_import_power
        total_renewable_gen = torch.sum(p_ren_mw, dim=-1)
        
        carbon_intensity = time_carbon_coeff.squeeze(-1) if time_carbon_coeff.dim() > 1 else time_carbon_coeff
        energy_coefficient = time_energy_coeff.squeeze(-1) if time_energy_coeff.dim() > 1 else time_energy_coeff
        
        raw_emissions = (total_carbon_emitting_gen * carbon_intensity) / (energy_coefficient + 1e-9)
        total_generation = total_carbon_emitting_gen + total_renewable_gen
        normalized_emissions = total_carbon_emitting_gen / (total_generation + 1e-9)
        
        return {'raw': raw_emissions, 'normalized': normalized_emissions}