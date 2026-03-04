"""
Physics-Informed Loss Functions for AC Optimal Power Flow (ACOPF)

Implements three differentiable constraint equations as PyTorch loss terms:

1. Power Balance (Equality)  — Eq. 3.8:  P_calc == P_true, Q_calc == Q_true
2. Voltage Limits (Inequality) — Eq. 3.10: V_min <= V_pred <= V_max
3. Branch Capacity (Inequality) — Eq. 3.11: |S_k| <= S_k_max

All losses use TRUE target values (not noisy features) as the reference,
per the physics regularization paradigm (Owerko et al., 2020).
"""

import torch
from torch import nn, Tensor


class PhysicsLoss(nn.Module):
    """
    Combined physics-informed loss for ACOPF constraints.

    Accepts predicted VM/VA, true targets, and grid parameters (Ybus, branch data).
    Returns a weighted sum of the three constraint violation penalties.

    Args:
        ybus:           Complex admittance matrix [N, N] (per-unit, already normalized by S_base)
        branch_from:    From-bus indices [L] (int64)
        branch_to:      To-bus indices [L] (int64)
        branch_max_s_pu: Thermal limit in per-unit apparent power [L] (float32)
        v_min:          Minimum voltage bound (p.u.), e.g. 0.85
        v_max:          Maximum voltage bound (p.u.), e.g. 1.15
        lambda_power:   Weight for power balance loss
        lambda_voltage: Weight for voltage limit loss
        lambda_branch:  Weight for branch capacity loss
    """

    def __init__(
        self,
        ybus: Tensor,
        branch_from: Tensor,
        branch_to: Tensor,
        branch_max_s_pu: Tensor,
        v_min: float = 0.90,
        v_max: float = 1.10,
        lambda_power: float = 0.1,
        lambda_voltage: float = 0.01,
        lambda_branch: float = 0.01,
    ):
        super().__init__()
        # Register as buffers so they move to GPU with the model
        self.register_buffer("ybus", ybus)                         # [N, N] complex128
        self.register_buffer("branch_from", branch_from)           # [L] int64
        self.register_buffer("branch_to", branch_to)               # [L] int64
        self.register_buffer("branch_max_s_pu", branch_max_s_pu)   # [L] float32

        self.v_min = v_min
        self.v_max = v_max
        self.lambda_power = lambda_power
        self.lambda_voltage = lambda_voltage
        self.lambda_branch = lambda_branch

    def _reconstruct_complex_voltage(self, vm_pred: Tensor, va_pred: Tensor) -> Tensor:
        """
        Reconstruct complex voltage phasor from predicted VM and VA.

        Our preprocessing centered VM by subtracting 1.0 (nominal),
        so we must add 1.0 back to get absolute per-unit voltage.

        Args:
            vm_pred: Predicted voltage magnitude deviation [B, N] (centered around 0)
            va_pred: Predicted voltage angle in radians [B, N]

        Returns:
            Complex voltage vector [B, N] as complex64/128
        """
        vm_abs = vm_pred + 1.0  # Restore from deviation to absolute p.u.
        
        # Ensure we use complex128 to match Ybus precision
        v_complex = vm_abs.to(self.ybus.dtype) * torch.exp(1j * va_pred.to(self.ybus.dtype))
        return v_complex

    def power_balance_loss(self, v_complex: Tensor, targets: Tensor) -> Tensor:
        """
        AC Power Flow Equation (Eq. 3.8) — Equality Constraint.

        Computes: S_calc = diag(V) * conj(Ybus @ V)
        Then:     P_calc = Re(S_calc),  Q_calc = Im(S_calc)
        Loss:     MSE(P_calc - P_net_true) + MSE(Q_calc - Q_net_true)

        The true net power injection at each bus:
            P_net = P_ext_grid + P_conv + P_ren - P_load
            Q_net = Q_ext_grid + Q_conv + Q_ren - Q_load

        Args:
            v_complex: Complex voltage phasor [B, N]
            targets:   Full target tensor [B, N, 10] (all 10 target columns) or flat [B*N, 10]

        Returns:
            Scalar loss
        """
        # S_calc = V * conj(Ybus @ V^T)
        # Ybus is [N, N], v_complex is [B, N]
        # Use matmul with unsqueeze to handle batching properly: [N, N] @ [B, N, 1] -> [B, N, 1]
        i_injected = torch.matmul(self.ybus, v_complex.unsqueeze(-1)).squeeze(-1)  # [B, N]
        s_calc = v_complex * torch.conj(i_injected)  # [B, N]

        p_calc = s_calc.real.float()
        q_calc = s_calc.imag.float()

        # Reshape targets if they were passed flattened (e.g. from spatial models)
        if targets.dim() == 2:
            targets = targets.reshape(v_complex.shape[0], v_complex.shape[1], -1)

        # Target indices: P_LOAD=0, Q_LOAD=1, P_EXT=2, Q_EXT=3,
        #                 P_CONV=4, Q_CONV=5, P_REN=6, Q_REN=7
        p_net_true = (
            targets[..., 2]     # P_ext_grid
            + targets[..., 4]   # P_conv
            + targets[..., 6]   # P_ren
            - targets[..., 0]   # P_load
        )
        q_net_true = (
            targets[..., 3]     # Q_ext_grid
            + targets[..., 5]   # Q_conv
            + targets[..., 7]   # Q_ren
            - targets[..., 1]   # Q_load
        )

        loss_p = torch.mean((p_calc - p_net_true) ** 2)
        loss_q = torch.mean((q_calc - q_net_true) ** 2)

        return loss_p + loss_q

    def voltage_limit_loss(self, vm_pred: Tensor) -> Tensor:
        """
        Voltage Magnitude Constraint (Eq. 3.10) — Inequality Constraint.

        Penalizes predictions outside [V_min, V_max] using ReLU:
            Loss = MSE(ReLU(V - V_max)) + MSE(ReLU(V_min - V))

        Args:
            vm_pred: Predicted voltage magnitude deviation [B, N] (centered around 0)

        Returns:
            Scalar loss
        """
        vm_abs = vm_pred + 1.0  # Restore absolute p.u.

        over = torch.relu(vm_abs - self.v_max)
        under = torch.relu(self.v_min - vm_abs)

        return torch.mean(over ** 2) + torch.mean(under ** 2)

    def branch_capacity_loss(self, v_complex: Tensor) -> Tensor:
        """
        Branch Thermal Capacity Constraint (Eq. 3.11) — Inequality Constraint.

        For each branch k connecting from_bus → to_bus:
            I_k = Y_branch_k * (V_from - V_to)
            S_k = |V_from * conj(I_k)|

        We extract the branch admittance directly from the Ybus off-diagonal:
            Y_branch_k = -Ybus[from, to]  (negative of off-diagonal element)

        Computes apparent power flow and penalizes if |S_k| > S_k_max.

        Note: S_k_max is derived from max_i_ka using the approximation:
              S_max ≈ V_nom * I_max (in per-unit, V_nom ≈ 1.0)
              So S_max ≈ max_i_ka (when working in per-unit with V_base = 1.0)

        For case33bw, max_i_ka = 99999, so this loss will be ~0 (unconstrained).
        For case57/case118, real thermal limits will activate this constraint.

        Args:
            v_complex: Complex voltage phasor [B, N]

        Returns:
            Scalar loss
        """
        f_idx = self.branch_from  # [L]
        t_idx = self.branch_to    # [L]

        v_from = v_complex[:, f_idx]  # [B, L]
        v_to = v_complex[:, t_idx]    # [B, L]

        # Branch admittance from Ybus off-diagonal (negative sign convention)
        y_branch = -self.ybus[f_idx, t_idx]  # [L]

        # Branch current: I_k = Y_branch * (V_from - V_to)
        i_branch = y_branch.unsqueeze(0) * (v_from - v_to)  # [B, L]

        # Apparent power magnitude: |S_k| = |V_from * conj(I_k)|
        s_branch = torch.abs(v_from * torch.conj(i_branch))  # [B, L]

        # S_max: in per-unit
        s_max = self.branch_max_s_pu.unsqueeze(0).float()  # [1, L]

        violation = torch.relu(s_branch.float() - s_max)
        return torch.mean(violation ** 2)

    def forward(
        self,
        vm_pred: Tensor,
        va_pred: Tensor,
        targets: Tensor,
    ) -> dict:
        """
        Compute all three physics constraint losses.

        Args:
            vm_pred: Predicted VM deviation [B, N] or [B*N]
            va_pred: Predicted VA radians [B, N] or [B*N]
            targets: Full target tensor [B, N, 10] or [B*N, 10]

        Returns:
            Dictionary with individual and total weighted losses.
        """
        # Auto-reshape if flat (spatial models case)
        if vm_pred.dim() == 1:
            num_nodes = self.ybus.shape[0]
            batch_size = vm_pred.shape[0] // num_nodes
            vm_pred = vm_pred.reshape(batch_size, num_nodes)
            va_pred = va_pred.reshape(batch_size, num_nodes)

        v_complex = self._reconstruct_complex_voltage(vm_pred, va_pred)

        loss_power = self.power_balance_loss(v_complex, targets)
        loss_voltage = self.voltage_limit_loss(vm_pred)
        loss_branch = self.branch_capacity_loss(v_complex)

        total = (
            self.lambda_power * loss_power
            + self.lambda_voltage * loss_voltage
            + self.lambda_branch * loss_branch
        )

        return {
            "physics_loss": total,
            "power_balance_loss": loss_power.detach(),
            "voltage_limit_loss": loss_voltage.detach(),
            "branch_capacity_loss": loss_branch.detach(),
        }
