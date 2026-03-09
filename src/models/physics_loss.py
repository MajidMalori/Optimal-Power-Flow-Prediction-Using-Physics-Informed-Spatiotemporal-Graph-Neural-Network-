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
        contingencies: Tensor = None,
        v_min: float = 0.90,
        v_max: float = 1.10,
        lambda_power: float = 0.1,
        lambda_voltage: float = 0.01,
        lambda_branch: float = 0.01,
    ):
        super().__init__()
        # Register as buffers so they move to GPU with the model
        self.register_buffer("ybus_base", ybus)                    # [N, N] complex128
        self.register_buffer("branch_from", branch_from)           # [L] int64
        self.register_buffer("branch_to", branch_to)               # [L] int64
        self.register_buffer("branch_max_s_pu", branch_max_s_pu)   # [L] float32
        
        if contingencies is not None and contingencies.numel() > 0:
            self.register_buffer("contingencies", contingencies)   # [K, N, N]
        else:
            self.contingencies = None

        self.v_min = v_min
        self.v_max = v_max
        self.lambda_power = lambda_power
        self.lambda_voltage = lambda_voltage
        self.lambda_branch = lambda_branch

    def _reconstruct_complex_voltage(self, vm_pred: Tensor, va_pred: Tensor) -> Tensor:
        """
        Reconstruct complex voltage phasor from predicted VM and VA.
        """
        vm_abs = vm_pred + 1.0  # Restore from deviation to absolute p.u.
        v_complex = vm_abs.to(self.ybus_base.dtype) * torch.exp(1j * va_pred.to(self.ybus_base.dtype))
        return v_complex

    def power_balance_loss(self, v_complex: Tensor, targets: Tensor, topology_ids: Tensor) -> Tensor:
        """
        AC Power Flow Equation (Eq. 3.8) — Equality Constraint.
        Vectorized selection of correct Ybus per sample.
        """
        batch_size, num_nodes = v_complex.shape
        
        # Select correct Ybus for each sample
        # topology_ids == 0 -> use ybus_base
        # topology_ids > 0  -> use contingencies[topo_id - 1]
        
        # We'll use a loop here for stability with diverse topologies in one batch
        i_injected_list = []
        for b in range(batch_size):
            tid = topology_ids[b].item()
            if tid == 0 or self.contingencies is None:
                yb = self.ybus_base
            else:
                yb = self.contingencies[tid - 1]
            
            # [N, N] @ [N, 1] -> [N]
            i_b = torch.matmul(yb, v_complex[b].unsqueeze(-1)).squeeze(-1)
            i_injected_list.append(i_b)
            
        i_injected = torch.stack(i_injected_list) # [B, N]
        s_calc = v_complex * torch.conj(i_injected)  # [B, N]

        p_calc = s_calc.real.float()
        q_calc = s_calc.imag.float()

        # Reshape targets if they were passed flattened
        if targets.dim() == 2:
            targets = targets.reshape(batch_size, num_nodes, -1)

        p_net_true = (targets[..., 2] + targets[..., 4] + targets[..., 6] - targets[..., 0])
        q_net_true = (targets[..., 3] + targets[..., 5] + targets[..., 7] - targets[..., 1])

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

    def branch_capacity_loss(self, v_complex: Tensor, topology_ids: Tensor) -> Tensor:
        """
        Branch Thermal Capacity Constraint (Eq. 3.11) — Inequality Constraint.
        """
        batch_size = v_complex.shape[0]
        f_idx = self.branch_from  # [L]
        t_idx = self.branch_to    # [L]

        v_from = v_complex[:, f_idx]  # [B, L]
        v_to = v_complex[:, t_idx]    # [B, L]

        # Select correct y_branch per sample
        y_branch_list = []
        for b in range(batch_size):
            tid = topology_ids[b].item()
            if tid == 0 or self.contingencies is None:
                yb = self.ybus_base
            else:
                yb = self.contingencies[tid - 1]
            y_branch_list.append(-yb[f_idx, t_idx])
            
        y_branch = torch.stack(y_branch_list) # [B, L]

        # Branch current: I_k = Y_branch * (V_from - V_to)
        i_branch = y_branch * (v_from - v_to)  # [B, L]

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
        topology_ids: Tensor,
    ) -> dict:
        """
        Compute all three physics constraint losses.
        """
        # Auto-reshape if flat (spatial models case)
        if vm_pred.dim() == 1:
            num_nodes = self.ybus_base.shape[0]
            batch_size = vm_pred.shape[0] // num_nodes
            vm_pred = vm_pred.reshape(batch_size, num_nodes)
            va_pred = va_pred.reshape(batch_size, num_nodes)
        else:
            batch_size = vm_pred.shape[0]

        v_complex = self._reconstruct_complex_voltage(vm_pred, va_pred)

        loss_power = self.power_balance_loss(v_complex, targets, topology_ids)
        loss_voltage = self.voltage_limit_loss(vm_pred)
        loss_branch = self.branch_capacity_loss(v_complex, topology_ids)

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

    @torch.no_grad()
    def evaluate_constraints(
        self,
        vm_pred: Tensor,
        va_pred: Tensor,
        targets: Tensor,
        topology_ids: Tensor,
        p_tol: float = 0.005,  # 0.5% power balance tolerance (standard for OPF)
    ) -> dict:
        """
        Evaluate binary constraint satisfaction for benchmarking.
        Returns counts of violations and total items checked.
        """
        # Auto-reshape if flat
        if vm_pred.dim() == 1:
            num_nodes = self.ybus_base.shape[0]
            batch_size = vm_pred.shape[0] // num_nodes
            vm_pred = vm_pred.reshape(batch_size, num_nodes)
            va_pred = va_pred.reshape(batch_size, num_nodes)
        else:
            batch_size, num_nodes = vm_pred.shape

        v_complex = self._reconstruct_complex_voltage(vm_pred, va_pred)
        
        # 1. Power Balance
        i_injected_list = []
        for b in range(batch_size):
            tid = topology_ids[b].item()
            yb = self.ybus_base if (tid == 0 or self.contingencies is None) else self.contingencies[tid - 1]
            i_injected_list.append(torch.matmul(yb, v_complex[b].unsqueeze(-1)).squeeze(-1))
        
        i_injected = torch.stack(i_injected_list)
        s_calc = v_complex * torch.conj(i_injected)
        p_calc = s_calc.real.float()
        q_calc = s_calc.imag.float()

        if targets.dim() == 2:
            targets = targets.reshape(batch_size, num_nodes, -1)

        p_net_true = (targets[..., 2] + targets[..., 4] + targets[..., 6] - targets[..., 0])
        q_net_true = (targets[..., 3] + targets[..., 5] + targets[..., 7] - targets[..., 1])

        p_err = torch.abs(p_calc - p_net_true)
        q_err = torch.abs(q_calc - q_net_true)
        
        p_satisfied = (p_err < p_tol).sum().item()
        q_satisfied = (q_err < p_tol).sum().item()
        total_p_q = batch_size * num_nodes

        # 2. Voltage Limits
        vm_abs = vm_pred + 1.0
        v_satisfied = ((vm_abs >= self.v_min) & (vm_abs <= self.v_max)).sum().item()
        total_v = batch_size * num_nodes

        # 3. Branch Capacity
        f_idx = self.branch_from
        t_idx = self.branch_to
        v_from = v_complex[:, f_idx]
        v_to = v_complex[:, t_idx]
        
        y_branch_list = []
        for b in range(batch_size):
            tid = topology_ids[b].item()
            yb = self.ybus_base if (tid == 0 or self.contingencies is None) else self.contingencies[tid - 1]
            y_branch_list.append(-yb[f_idx, t_idx])
            
        y_branch = torch.stack(y_branch_list)
        s_branch = torch.abs(v_from * torch.conj(y_branch * (v_from - v_to)))
        s_max = self.branch_max_s_pu.unsqueeze(0).float()
        
        s_satisfied = (s_branch <= s_max).sum().item()
        total_s = batch_size * s_branch.shape[1]

        # A sample is fully "feasible" if ALL its constraints are met
        p_v_s_all_met = (p_err < p_tol).all(dim=1) & (q_err < p_tol).all(dim=1) & \
                        ((vm_abs >= self.v_min) & (vm_abs <= self.v_max)).all(dim=1) & \
                        (s_branch <= s_max).all(dim=1)
        feasible_samples = p_v_s_all_met.sum().item()

        return {
            "p_satisfied": p_satisfied,
            "q_satisfied": q_satisfied,
            "v_satisfied": v_satisfied,
            "s_satisfied": s_satisfied,
            "total_p_q": total_p_q,
            "total_v": total_v,
            "total_s": total_s,
            "feasible_samples": feasible_samples,
            "total_samples": batch_size
        }
