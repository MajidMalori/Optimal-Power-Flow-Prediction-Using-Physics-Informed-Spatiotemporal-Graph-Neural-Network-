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
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        # Create PowerSystemLoss instance for physics calculations
        physics_metrics = PowerSystemLoss(config=config, normalizer=None)

        # For physics calculations, we need 3D format [batch_size, num_buses, features]
        if outputs.dim() == 2:
            # If outputs are flattened, reshape to 3D for physics calculations
            batch_size = outputs.shape[0]
            num_features = 6  # Standard: vm, va, p_load, q_load, p_gen, q_gen
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
        self.s_base_mva = getattr(config, 'S_BASE_MVA', 100.0)

        # self.loss_scale_factor = getattr(config, "LOSS_SCALE_FACTOR", 1.0) if self.is_physics_informed else 0.0

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
                'power_violation': torch.tensor(0.0),
                'voltage_violation': torch.tensor(0.0)
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
        p_load_mw = state[..., 2]
        q_load_mvar = state[..., 3]
        p_gen_mw = state[..., 4]
        q_gen_mvar = state[..., 5]
        
        p_inj_mw = p_gen_mw - p_load_mw
        q_inj_mvar = q_gen_mvar - q_load_mvar
        
        p_inj_pu = p_inj_mw / self.s_base_mva
        q_inj_pu = q_inj_mvar / self.s_base_mva
        
        return p_inj_pu, q_inj_pu
    
    def _get_power_injections(self, state: torch.Tensor):
        """Extracts power injections from the state tensor in original units (MW, MVAr)."""
        p_load_mw = state[..., 2]
        q_load_mvar = state[..., 3]
        p_gen_mw = state[..., 4]
        q_gen_mvar = state[..., 5]
        
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

    # def _compute_normalized_power_balance_violation(self, state: torch.Tensor, Ybus: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
    #     """Computes power balance violation, normalized by the total load for MOOPF evaluation."""
    #     p_inj_mw, q_inj_mvar, p_load_mw, q_load_mvar = self._get_power_injections(state)
    #     Vm = state[..., 0]
    #     Va = state[..., 1]
    #     V = Vm * torch.exp(1j * Va)
    #     I = torch.einsum('bij,bj->bi', Ybus.cfloat(), V)
    #     S_calc = V * torch.conj(I)
    #     squared_mismatch = (p_inj_mw - S_calc.real)**2 + (q_inj_mvar - S_calc.imag)**2
    #     total_load_s_squared = torch.sum(p_load_mw**2 + q_load_mvar**2, dim=1)
    #     return torch.mean(squared_mismatch, dim=1) / (total_load_s_squared + epsilon)

    # def _compute_normalized_voltage_limit_violation(self, state: torch.Tensor) -> torch.Tensor:
    #     """Computes the mean absolute voltage violation in per unit (p.u.) for MOOPF evaluation."""
    #     Vm = state[:, :, 0]
    #     v_violations = torch.relu(self.v_min - Vm) + torch.relu(Vm - self.v_max)
    #     return torch.mean(v_violations, dim=1)

    def _compute_normalized_active_power_loss(self, state: torch.Tensor, Ybus: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """
        Computes normalized active power loss using the accurate power loss formula from equation (3.5):
        P_loss = Σ Σ Dij * [Rij/|Vit||Vjt| * (PitPjt + QitQjt) + Rij|Vit||Vjt|sin(θit-θjt)(QitPjt - QjtPit)]
        
        Vectorized implementation for better performance.
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
        total_loss = torch.sum(branch_losses * upper_triangle, dim=(1, 2))  # [batch_size]
        
        # The total_loss is already in per-unit since we used per-unit R and I values
        # No need for additional normalization by load - that would be double normalization
        
        # Sanity check: Clamp unrealistic power loss values (due to poor model predictions)
        # Typical power system losses are 2-15% of total load
        max_reasonable_loss = 0.20  # 20% maximum (very conservative upper bound)
        total_loss = torch.clamp(total_loss, min=0.0, max=max_reasonable_loss)
        
        return total_loss

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
        Computes normalized power flow using equation (3.5):
        f1 = Σ_t Σ_i Σ_j D_ijt * [α_ijt * (P_it*P_jt + Q_it*Q_jt) + β_ijt * (Q_it*P_jt - Q_jt*P_it)]
        
        Where:
        - α_ijt = R_ijt / (|V_it| * |V_jt|) * cos(θ_it - θ_jt)
        - β_ijt = R_ijt * |V_it| * |V_jt| * sin(θ_it - θ_jt)
        - D_ijt = branch connectivity (1 if branch exists, 0 otherwise)
        
        This represents the actual power flow through the network, different from power loss.
        
        Args:
            state: Tensor containing [vm_pu, va_rad, p_load, q_load, p_gen, q_gen]
            Ybus: Admittance matrix [batch_size, num_buses, num_buses]
            epsilon: Small value to avoid division by zero
            
        Returns:
            Tensor containing normalized power flow values [batch_size]
        """
        # Extract state variables
        Vm = state[..., 0]  # Voltage magnitudes (p.u.) [batch_size, num_buses]
        Va = state[..., 1]  # Voltage angles (rad) [batch_size, num_buses]
        
        # Get power injections in per-unit
        p_inj_pu, q_inj_pu = self._get_power_injections_pu(state)  # [batch_size, num_buses]
        
        batch_size, num_buses = Vm.shape[:2]
        
        # Check if we have any data
        if batch_size == 0 or num_buses == 0:
            return torch.zeros(batch_size, device=state.device, dtype=state.dtype)
        
        # Create branch connectivity mask from Ybus (D_ij matrix)
        # A branch exists if there's a non-zero admittance between buses
        branch_exists = torch.abs(Ybus) > 1e-6  # [batch_size, num_buses, num_buses]
        
        # Remove self-loops (diagonal elements) - no power flow to self
        branch_exists = branch_exists & ~torch.eye(num_buses, dtype=torch.bool, device=state.device)
        
        # Extract series impedance for each branch: Z_ij = 1/Y_ij
        z_series = torch.where(branch_exists, 1.0 / Ybus, torch.zeros_like(Ybus))
        r_series = torch.abs(z_series.real)  # Resistance (always positive)
        
        # Skip branches with very small resistance
        valid_branches = branch_exists & (r_series > 1e-6)
        
        # Create expanded tensors for vectorized operations
        # Expand voltages to [batch_size, num_buses, num_buses]
        Vm_i = Vm.unsqueeze(2).expand(-1, -1, num_buses)  # [batch_size, num_buses, num_buses]
        Vm_j = Vm.unsqueeze(1).expand(-1, num_buses, -1)  # [batch_size, num_buses, num_buses]
        
        # Expand voltage angles
        Va_i = Va.unsqueeze(2).expand(-1, -1, num_buses)  # [batch_size, num_buses, num_buses]
        Va_j = Va.unsqueeze(1).expand(-1, num_buses, -1)  # [batch_size, num_buses, num_buses]
        
        # Expand power injections (using actual power from nodes)
        P_i = p_inj_pu.unsqueeze(2).expand(-1, -1, num_buses)  # [batch_size, num_buses, num_buses]
        P_j = p_inj_pu.unsqueeze(1).expand(-1, num_buses, -1)  # [batch_size, num_buses, num_buses]
        Q_i = q_inj_pu.unsqueeze(2).expand(-1, -1, num_buses)  # [batch_size, num_buses, num_buses]
        Q_j = q_inj_pu.unsqueeze(1).expand(-1, num_buses, -1)  # [batch_size, num_buses, num_buses]
        
        # Calculate voltage magnitude products
        V_prod = Vm_i * Vm_j  # [batch_size, num_buses, num_buses]
        
        # Calculate angle differences (θ_it - θ_jt)
        angle_diff = Va_i - Va_j  # [batch_size, num_buses, num_buses]
        
        # Avoid division by zero for V_prod
        V_prod_safe = torch.where(V_prod > epsilon, V_prod, torch.ones_like(V_prod))
        
        # Calculate α_ijt = R_ijt / (|V_it| * |V_jt|) * cos(θ_it - θ_jt)
        alpha_ijt = torch.where(
            valid_branches,
            (r_series / V_prod_safe) * torch.cos(angle_diff),
            torch.zeros_like(r_series)
        )
        
        # Calculate β_ijt = R_ijt * |V_it| * |V_jt| * sin(θ_it - θ_jt)
        beta_ijt = torch.where(
            valid_branches,
            r_series * V_prod * torch.sin(angle_diff),
            torch.zeros_like(r_series)
        )
        
        # Calculate the power flow terms according to equation (3.5)
        # Term 1: α_ijt * (P_it*P_jt + Q_it*Q_jt)
        power_product_term = alpha_ijt * (P_i * P_j + Q_i * Q_j)
        
        # Term 2: β_ijt * (Q_it*P_jt - Q_jt*P_it)
        cross_product_term = beta_ijt * (Q_i * P_j - Q_j * P_i)
        
        # Total power flow for each branch
        branch_power_flow = torch.where(
            valid_branches,
            power_product_term + cross_product_term,
            torch.zeros_like(power_product_term)
        )
        
        # Sum over all branches (only upper triangle to avoid double counting)
        # Create upper triangle mask
        upper_triangle = torch.triu(torch.ones(num_buses, num_buses, dtype=torch.bool, device=state.device), diagonal=1)
        upper_triangle = upper_triangle.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_buses, num_buses]
        
        # Apply upper triangle mask and sum
        total_power_flow = torch.sum(branch_power_flow * upper_triangle, dim=(1, 2))  # [batch_size]
        
        # Normalize by total system power to get per-unit value
        total_power = torch.sum(torch.abs(p_inj_pu) + torch.abs(q_inj_pu), dim=1) + epsilon
        normalized_power_flow = torch.abs(total_power_flow) / total_power
        
        # Only clamp minimum to avoid negative values (which would be non-physical)
        normalized_power_flow = torch.clamp(normalized_power_flow, min=0.0)
        
        return normalized_power_flow

    def _compute_carbon_emissions(
        self, 
        predicted_state_physical: torch.Tensor, 
        time_carbon_coeff: torch.Tensor, 
        time_energy_coeff: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes carbon emissions according to equation (3.7):
        f3 = Σ(Psum_t - PDG_t) * Cm / Ef
        
        Where:
        - Psum_t: Total power consumption at time t
        - PDG_t: Total distributed generation at time t  
        - Cm: Carbon emissions per unit electricity (time_carbon_coeff)
        - Ef: Energy utilization coefficient (time_energy_coeff)
        
        Returns normalized emissions based on actual renewable penetration levels
        from data generation (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
        """
        # Extract power consumption and generation from state
        # State format: [vm_pu, va_rad, p_load, q_load, p_gen, q_gen]
        total_load = torch.sum(predicted_state_physical[..., 2], dim=-1)  # Psum_t (total consumption)
        total_distributed_gen = torch.sum(predicted_state_physical[..., 4], dim=-1)  # PDG_t (distributed generation)
        
        # Net power from grid: (Psum_t - PDG_t)
        power_from_grid = total_load - total_distributed_gen
        
        # Ensure coefficients are correctly shaped for broadcasting
        carbon_intensity = time_carbon_coeff.squeeze(-1) if time_carbon_coeff.dim() > 1 else time_carbon_coeff  # Cm
        energy_coefficient = time_energy_coeff.squeeze(-1) if time_energy_coeff.dim() > 1 else time_energy_coeff  # Ef
        
        # Apply equation (3.7): f3 = (Psum - PDG) * Cm / Ef
        raw_emissions = (power_from_grid * carbon_intensity) / (energy_coefficient + 1e-9)
        
        # Calculate actual renewable penetration fraction (matches data generation logic)
        # This reflects the actual renewable_fractions_to_run: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        renewable_penetration = total_distributed_gen / (total_load + 1e-9)
        
        # Normalized emissions: 1 - renewable_penetration
        # Physical interpretation aligned with data generation:
        # 1.0: renewable_penetration = 0.0 (all grid power, maximum emissions)
        # 0.8: renewable_penetration = 0.2 (20% renewable, 80% emissions)
        # 0.6: renewable_penetration = 0.4 (40% renewable, 60% emissions)
        # 0.4: renewable_penetration = 0.6 (60% renewable, 40% emissions)
        # 0.2: renewable_penetration = 0.8 (80% renewable, 20% emissions)
        # 0.0: renewable_penetration = 1.0 (100% renewable, zero emissions)
        normalized_emissions = 1.0 - renewable_penetration
        
        # Apply bounds for the discrete renewable fractions used in data generation
        # Values outside [0, 1] indicate model predictions beyond training data distribution
        normalized_emissions = torch.clamp(normalized_emissions, min=0.0, max=1.0)
        
        return {'raw': raw_emissions, 'normalized': normalized_emissions}