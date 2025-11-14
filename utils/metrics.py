import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, ybus_batch: torch.Tensor, config: object, bus_types: torch.Tensor) -> Dict[str, float]:
    """
    Computes both standard regression metrics and power system specific metrics.
    
    Args:
        outputs: Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
        targets: True unknowns [batch, buses, 2] (OPF: bus-type dependent)
        ybus_batch: Ybus matrices for physics calculations
        config: Configuration object
        bus_types: Required [batch, buses] with codes [0=PQ, 1=PV, 2=Slack] for OPF-specific metrics
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
        
        # OPF: Bus-type-specific metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
        }
        
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
    A comprehensive, physics-informed loss function for power system OPF.
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
        
        # Heteroscedastic vs Homoscedastic uncertainty
        self.use_heteroscedastic = getattr(config, 'USE_HETEROSCEDASTIC_UNCERTAINTY', False)
        
        if self.use_heteroscedastic:
            # Heteroscedastic: Model will predict natural parameters for DATA loss (MSE)
            # Model should output [batch, buses, 4]: [η1_var1, η1_var2, f2_var1, f2_var2]
            # where η1 = f1 (direct), η2 = -g+(f2) with g+ being exp or softplus
            # 
            # For PHYSICS losses: Use Kendall-style learnable weights (not heteroscedastic)
            # Heteroscedastic paper (Immer et al. 2023) only addresses data loss, not physics losses
            # Kendall et al. (CVPR 2018) provides well-cited method for multi-task loss weighting
            self.log_sigma_data = None  # Data loss uses heteroscedastic (natural parametrization)
            self.log_sigma_power = nn.Parameter(torch.tensor(0.0))  # Physics loss: Kendall-style
            self.log_sigma_voltage = nn.Parameter(torch.tensor(0.0))  # Physics loss: Kendall-style
            
            # Natural Parametrization (Immer et al., NeurIPS 2023)
            # "Effective Bayesian Heteroscedastic Regression with Deep Neural Networks"
            # Paper: https://arxiv.org/abs/2306.17758, Citations: 27+ (as of Nov 2025)
            # 
            # Natural parameters: η1 = μ/σ², η2 = -1/(2σ²) < 0
            # Key advantages:
            # 1. Jointly concave objective (more stable optimization)
            # 2. Simpler gradients: ∇η1 = μ - y, ∇η2 = σ² - (y² - μ²)
            # 3. No negative log variance issues (η2 < 0 by construction)
            self.loss_type = 'natural'
            natural_function = getattr(config, 'HETEROSCEDASTIC_NATURAL_FUNCTION', 'exp')
            print(f"[Heteroscedastic Loss] Using: Natural Parametrization (Immer et al., NeurIPS 2023) with {natural_function}")
        else:
            # Homoscedastic: Learnable uncertainty parameters (single value per loss term)
            self.log_sigma_data = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_power = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_voltage = nn.Parameter(torch.tensor(0.0))
            self.loss_type = None
        
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
                bus_types: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        
        # Ensure outputs and targets have the same shape
        if outputs_norm.dim() != targets_norm.dim():
            if outputs_norm.dim() == 2 and targets_norm.dim() == 3:
                # Reshape flattened 2D output [batch, buses*features] to 3D [batch, buses, features]
                outputs_norm = outputs_norm.view(targets_norm.shape)
            else:
                 raise ValueError(f"Shape mismatch: outputs {outputs_norm.shape}, targets {targets_norm.shape}")
        
        # Handle heteroscedastic vs homoscedastic output shapes
        if self.use_heteroscedastic:
            # Heteroscedastic: Model outputs [batch, buses, 4] = [var1_pred, var2_pred, log_sigma_var1, log_sigma_var2]
            expected_features = 4
        else:
            # Homoscedastic: Model outputs [batch, buses, 2] = [var1_pred, var2_pred]
            expected_features = 2
        
        # Ensure outputs are in correct format
        if outputs_norm.dim() == 2:
            # Flattened output: reshape to [batch, buses, expected_features]
            batch_size = outputs_norm.shape[0]
            num_buses = targets_norm.shape[1]  # Get num_buses from targets
            expected_size = batch_size * num_buses * expected_features
            actual_size = outputs_norm.numel()
            if actual_size != expected_size:
                raise ValueError(
                    f"Cannot reshape outputs from {outputs_norm.shape} (size={actual_size}) "
                    f"to [batch={batch_size}, buses={num_buses}, {expected_features}] (expected size={expected_size}). "
                    f"Targets shape: {targets_norm.shape}, Heteroscedastic: {self.use_heteroscedastic}"
                )
            outputs_norm = outputs_norm.view(batch_size, num_buses, expected_features)
        elif outputs_norm.dim() == 3:
            # Already 3D: check if it's correct shape
            batch_size, num_buses = outputs_norm.shape[0], outputs_norm.shape[1]
            total_features = outputs_norm.shape[-1]
            actual_size = outputs_norm.numel()
            expected_size = batch_size * num_buses * expected_features
            
            if total_features == expected_features:
                # Correct shape
                pass  # Already correct
            elif total_features == num_buses * expected_features:
                # Wrong shape: [batch, buses, buses*features] - this is a bug
                raise ValueError(
                    f"Model output has wrong shape: {outputs_norm.shape}. "
                    f"Each bus has {total_features} features (should be {expected_features}). "
                    f"This suggests the output layer is producing [batch, buses, buses*{expected_features}] instead of [batch, buses, {expected_features}]. "
                    f"Actual size: {actual_size}, Expected: {expected_size}. "
                    f"Targets shape: {targets_norm.shape}, Heteroscedastic: {self.use_heteroscedastic}"
                )
            else:
                raise ValueError(
                    f"Unexpected output shape: {outputs_norm.shape} (size={actual_size}). "
                    f"Expected [batch, buses, {expected_features}] (size={expected_size}). "
                    f"Got last dim {total_features}, expected {expected_features}. "
                    f"Targets shape: {targets_norm.shape}, Heteroscedastic: {self.use_heteroscedastic}"
                )
        
        # Extract natural parameters and compute loss
        if self.use_heteroscedastic:
            # Heteroscedastic: outputs_norm is [batch, buses, 4] = [η1_var1, η1_var2, f2_var1, f2_var2]
            # Natural parameters: η1 = μ/σ², η2 = -1/(2σ²) < 0
            # where η1 = f1 (direct), η2 = -g+(f2) with g+ being exp or softplus
            
            eta1_var1_raw = outputs_norm[..., 0]  # [batch, buses] - η1 for variable 1
            eta1_var2_raw = outputs_norm[..., 1]  # [batch, buses] - η1 for variable 2
            f2_var1_raw = outputs_norm[..., 2]    # [batch, buses] - f2 for variable 1
            f2_var2_raw = outputs_norm[..., 3]    # [batch, buses] - f2 for variable 2
            
            # Get natural function type (exp or softplus)
            natural_function = getattr(self.config, 'HETEROSCEDASTIC_NATURAL_FUNCTION', 'exp')
            softplus_beta = getattr(self.config, 'HETEROSCEDASTIC_SOFTPLUS_BETA', 1.0)
            
            # DEBUG: Check raw model outputs - ONLY PRINT WHEN NaN DETECTED
            if torch.isnan(eta1_var1_raw).any() or torch.isnan(f2_var1_raw).any():
                print(f"\n{'='*80}")
                print(f"[DEBUG] NaN in raw model outputs!")
                print(f"{'='*80}")
                print(f"  eta1_var1_raw: nan={torch.isnan(eta1_var1_raw).any().item()}, inf={torch.isinf(eta1_var1_raw).any().item()}, min={eta1_var1_raw.min().item():.6f}, max={eta1_var1_raw.max().item():.6f}")
                print(f"  f2_var1_raw: nan={torch.isnan(f2_var1_raw).any().item()}, inf={torch.isinf(f2_var1_raw).any().item()}, min={f2_var1_raw.min().item():.6f}, max={f2_var1_raw.max().item():.6f}")
                print(f"  eta1_var2_raw: nan={torch.isnan(eta1_var2_raw).any().item()}, inf={torch.isinf(eta1_var2_raw).any().item()}, min={eta1_var2_raw.min().item():.6f}, max={eta1_var2_raw.max().item():.6f}")
                print(f"  f2_var2_raw: nan={torch.isnan(f2_var2_raw).any().item()}, inf={torch.isinf(f2_var2_raw).any().item()}, min={f2_var2_raw.min().item():.6f}, max={f2_var2_raw.max().item():.6f}")
                print(f"{'='*80}")
                import sys
                sys.exit(1)
            
            # Compute g+(f2) - positive function ensuring η2 < 0
            if natural_function == 'exp':
                # g+(x) = 0.5 * exp(x) (from paper, Equation 4)
                g_plus_var1 = 0.5 * torch.exp(f2_var1_raw)
                g_plus_var2 = 0.5 * torch.exp(f2_var2_raw)
            elif natural_function == 'softplus':
                # g+(x) = (1/β) * log(1 + exp(β*x))
                g_plus_var1 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var1_raw)
                g_plus_var2 = (1.0 / softplus_beta) * F.softplus(softplus_beta * f2_var2_raw)
            else:
                raise ValueError(f"Unknown natural function: {natural_function}. Must be 'exp' or 'softplus'")
            
            # DEBUG: Check g+ values - ONLY PRINT WHEN NaN DETECTED
            if torch.isnan(g_plus_var1).any() or torch.isinf(g_plus_var1).any():
                print(f"\n{'='*80}")
                print(f"[DEBUG] NaN/Inf in g_plus_var1!")
                print(f"{'='*80}")
                print(f"  f2_var1_raw range: [{f2_var1_raw.min().item():.6f}, {f2_var1_raw.max().item():.6f}]")
                print(f"  f2_var2_raw range: [{f2_var2_raw.min().item():.6f}, {f2_var2_raw.max().item():.6f}]")
                print(f"  g_plus_var1 range: [{g_plus_var1.min().item():.6f}, {g_plus_var1.max().item():.6f}], nan={torch.isnan(g_plus_var1).any().item()}, inf={torch.isinf(g_plus_var1).any().item()}")
                print(f"  g_plus_var2 range: [{g_plus_var2.min().item():.6f}, {g_plus_var2.max().item():.6f}], nan={torch.isnan(g_plus_var2).any().item()}, inf={torch.isinf(g_plus_var2).any().item()}")
                print(f"  exp(f2_var1_raw) overflow: {(f2_var1_raw > 10).sum().item()} values > 10")
                print(f"  exp(f2_var2_raw) overflow: {(f2_var2_raw > 10).sum().item()} values > 10")
                print(f"{'='*80}")
                import sys
                sys.exit(1)
            
            # Compute natural parameters: η1 = f1 (direct), η2 = -g+(f2) < 0
            eta1_var1 = eta1_var1_raw  # [batch, buses]
            eta1_var2 = eta1_var2_raw  # [batch, buses]
            eta2_var1 = -g_plus_var1  # [batch, buses] - guaranteed < 0
            eta2_var2 = -g_plus_var2  # [batch, buses] - guaranteed < 0
            
            # Convert to mean and variance for predictions and MSE calculation
            # μ = -η1/(2η2), σ² = -1/(2η2)
            # Paper (Equation 3): No epsilon used - η2 < 0 by construction, so division is safe
            mu_var1 = -eta1_var1 / (2.0 * eta2_var1)  # [batch, buses]
            mu_var2 = -eta1_var2 / (2.0 * eta2_var2)  # [batch, buses]
            sigma2_var1 = -1.0 / (2.0 * eta2_var1)   # [batch, buses] - variance
            sigma2_var2 = -1.0 / (2.0 * eta2_var2)   # [batch, buses] - variance
            
            # NOTE: Paper does NOT use clamping (see Appendix E.2)
            # However, we add clamping for numerical stability in practice:
            # - Prevents division by zero when η2 → 0 (variance → ∞)
            # - Prevents overflow when η2 → -∞ (variance → 0)
            # - Ensures variance stays in reasonable range for power systems
            # 
            # If you want to match the paper exactly, set HETEROSCEDASTIC_USE_CLAMPING=False
            use_clamping = getattr(self.config, 'HETEROSCEDASTIC_USE_CLAMPING', False)  # Default: False to match paper
            if use_clamping:
                # Hardcoded defaults (not in paper, but useful for numerical stability)
                min_sigma = 0.01  # Minimum standard deviation
                max_sigma = 10.0  # Maximum standard deviation
                min_sigma2 = min_sigma ** 2
                max_sigma2 = max_sigma ** 2
                sigma2_var1 = sigma2_var1.clamp(min_sigma2, max_sigma2)
                sigma2_var2 = sigma2_var2.clamp(min_sigma2, max_sigma2)
            
            # Natural parametrization loss is computed on normalized targets
            # (since model outputs are in normalized space)
            var1_true_norm = targets_norm[..., 0]  # [batch, buses] - normalized
            var2_true_norm = targets_norm[..., 1]  # [batch, buses] - normalized
            
            # Compute natural parametrization negative log-likelihood (Equation 5 from paper)
            # log p(y|x,θ) = [η1, η2]ᵀ[y, y²]ᵀ + η1²/(4η2) + ½log(-2η2) + const
            # NLL = -log p(y|x,θ)
            # Note: This is computed on normalized targets (y_norm) since η1, η2 are in normalized space
            
            # For variable 1: y = var1_true_norm (normalized)
            y1 = var1_true_norm  # [batch, buses]
            y1_sq = y1 ** 2  # [batch, buses]
            
            # Inner product: [η1, η2]ᵀ[y, y²]ᵀ = η1*y + η2*y²
            inner1 = eta1_var1 * y1 + eta2_var1 * y1_sq  # [batch, buses]
            
            # Log partition: η1²/(4η2) + ½log(-2η2)
            # Note: η2 < 0, so -2η2 > 0, log(-2η2) is valid
            # Paper (Equation 5): No epsilon used
            log_partition1 = (eta1_var1 ** 2) / (4.0 * eta2_var1) + 0.5 * torch.log(-2.0 * eta2_var1)  # [batch, buses]
            
            # NLL for variable 1: -inner - log_partition (constant term omitted)
            nll_var1 = -inner1 - log_partition1  # [batch, buses]
            
            # For variable 2: y = var2_true_norm (normalized)
            y2 = var2_true_norm  # [batch, buses]
            y2_sq = y2 ** 2  # [batch, buses]
            
            inner2 = eta1_var2 * y2 + eta2_var2 * y2_sq  # [batch, buses]
            log_partition2 = (eta1_var2 ** 2) / (4.0 * eta2_var2) + 0.5 * torch.log(-2.0 * eta2_var2)  # [batch, buses]
            nll_var2 = -inner2 - log_partition2  # [batch, buses]
            
            # Total NLL: mean across batch and buses
            data_loss = torch.mean(nll_var1) + torch.mean(nll_var2)
            
            # DEBUG: Check for NaN in natural parametrization
            if torch.isnan(data_loss) or torch.isinf(data_loss):
                print(f"\n[DEBUG] NaN/Inf detected in data_loss!")
                print(f"  eta1_var1: min={eta1_var1.min().item():.6f}, max={eta1_var1.max().item():.6f}, nan={torch.isnan(eta1_var1).any().item()}")
                print(f"  eta2_var1: min={eta2_var1.min().item():.6f}, max={eta2_var1.max().item():.6f}, nan={torch.isnan(eta2_var1).any().item()}")
                print(f"  eta1_var2: min={eta1_var2.min().item():.6f}, max={eta1_var2.max().item():.6f}, nan={torch.isnan(eta1_var2).any().item()}")
                print(f"  eta2_var2: min={eta2_var2.min().item():.6f}, max={eta2_var2.max().item():.6f}, nan={torch.isnan(eta2_var2).any().item()}")
                print(f"  log_partition1: min={log_partition1.min().item():.6f}, max={log_partition1.max().item():.6f}, nan={torch.isnan(log_partition1).any().item()}")
                print(f"  log_partition2: min={log_partition2.min().item():.6f}, max={log_partition2.max().item():.6f}, nan={torch.isnan(log_partition2).any().item()}")
                print(f"  nll_var1: min={nll_var1.min().item():.6f}, max={nll_var1.max().item():.6f}, nan={torch.isnan(nll_var1).any().item()}")
                print(f"  nll_var2: min={nll_var2.min().item():.6f}, max={nll_var2.max().item():.6f}, nan={torch.isnan(nll_var2).any().item()}")
                print(f"  data_loss: {data_loss.item()}")
                # Check division by zero
                print(f"  eta2_var1 near zero: {(torch.abs(eta2_var1) < 1e-6).sum().item()} values")
                print(f"  eta2_var2 near zero: {(torch.abs(eta2_var2) < 1e-6).sum().item()} values")
                raise RuntimeError("NaN detected in natural parametrization loss - stopping training")
            
            # Compute MSE on NORMALIZED data for consistency with total loss (NLL is on normalized data)
            # This ensures MSE + p_vio + v_viol ≈ total_loss (all in same scale)
            mse_var1_norm = torch.mean((mu_var1 - var1_true_norm) ** 2)  # [batch, buses] -> scalar
            mse_var2_norm = torch.mean((mu_var2 - var2_true_norm) ** 2)  # [batch, buses] -> scalar
            unweighted_mse = mse_var1_norm + mse_var2_norm  # Normalized MSE (for consistency with total loss)
            
            # Also compute denormalized MSE for reporting (physical units)
            mu_norm = torch.stack([mu_var1, mu_var2], dim=-1)  # [batch, buses, 2]
            predictions_denorm = self.normalizer.denormalize(mu_norm)
            var1_pred = predictions_denorm[..., 0]
            var2_pred = predictions_denorm[..., 1]
            targets_denorm_for_mse = self.normalizer.denormalize(targets_norm)
            var1_true = targets_denorm_for_mse[..., 0]
            var2_true = targets_denorm_for_mse[..., 1]
            mse_var1_denorm = torch.mean((var1_pred - var1_true) ** 2)  # Denormalized (for reporting)
            mse_var2_denorm = torch.mean((var2_pred - var2_true) ** 2)  # Denormalized (for reporting)
            
            # Compute sigma (standard deviation) for reporting and violations
            # Note: sigma2 is in normalized space, but we report it as-is (relative uncertainty)
            if use_clamping:
                sigma_var1 = torch.sqrt(sigma2_var1.clamp(min_sigma2, max_sigma2))  # [batch, buses]
                sigma_var2 = torch.sqrt(sigma2_var2.clamp(min_sigma2, max_sigma2))  # [batch, buses]
            else:
                # No clamping: match paper exactly (Equation 3: σ² = -1/(2η2), so σ = sqrt(σ²))
                # Paper: No epsilon used
                sigma_var1 = torch.sqrt(sigma2_var1)  # [batch, buses]
                sigma_var2 = torch.sqrt(sigma2_var2)  # [batch, buses]
            
            # Store sigma tensors for use in violation calculations
            self._sigma_var1_for_violations = sigma_var1  # [batch, buses]
            self._sigma_var2_for_violations = sigma_var2  # [batch, buses]
        else:
            # Homoscedastic: outputs_norm is [batch, buses, 2]
            outputs_denorm_for_mse = self.normalizer.denormalize(outputs_norm)
            targets_denorm_for_mse = self.normalizer.denormalize(targets_norm)
            
            var1_pred = outputs_denorm_for_mse[..., 0]
            var2_pred = outputs_denorm_for_mse[..., 1]
            var1_true = targets_denorm_for_mse[..., 0]
            var2_true = targets_denorm_for_mse[..., 1]
            
            # Compute denormalized MSE (physical units)
            mse_var1_denorm = self.mse_loss_fn(var1_pred, var1_true)
            mse_var2_denorm = self.mse_loss_fn(var2_pred, var2_true)
            # For homoscedastic, data_loss is denormalized (used for optimization)
            data_loss = mse_var1_denorm + mse_var2_denorm
            unweighted_mse = data_loss  # For homoscedastic, unweighted = weighted
            # Store for later use in reporting
            mse_var1 = mse_var1_denorm
            mse_var2 = mse_var2_denorm
        
        # If not physics-informed, we are done.
        if not self.is_physics_informed:
            # ML Engineering Best Practice: Report denormalized MSE (physical units) for interpretability
            if self.use_heteroscedastic:
                mse_denorm_val = mse_var1_denorm + mse_var2_denorm
                total_loss_denorm_val = mse_denorm_val  # For non-physics, total = MSE
                return {
                    'total_loss': data_loss,  # Normalized (for optimization)
                    'total_loss_denorm': total_loss_denorm_val,  # Denormalized (physical units - for display)
                    'mse': mse_denorm_val,  # Denormalized MSE (physical units - what ML engineers show)
                    'mse_normalized': unweighted_mse,  # Normalized MSE (for reference)
                    'mse_var1': mse_var1_norm,
                    'mse_var2': mse_var2_norm,
                    'power_violation': torch.tensor(0.0, device=data_loss.device),
                    'voltage_violation': torch.tensor(0.0, device=data_loss.device)
                }
            else:
                # For homoscedastic, mse_var1_denorm and mse_var2_denorm were already computed above
                mse_denorm_val = mse_var1_denorm + mse_var2_denorm
                total_loss_denorm_val = mse_denorm_val  # For non-physics, total = MSE
                return {
                    'total_loss': data_loss,  # For homoscedastic, data_loss is denormalized (no normalization)
                    'total_loss_denorm': total_loss_denorm_val,  # Denormalized (physical units - for display)
                    'mse': mse_denorm_val,  # Denormalized MSE (physical units - what ML engineers show)
                    'mse_normalized': data_loss,  # For homoscedastic, normalized = denormalized
                    'mse_var1': mse_var1,
                    'mse_var2': mse_var2,
                    'power_violation': torch.tensor(0.0, device=data_loss.device),
                    'voltage_violation': torch.tensor(0.0, device=data_loss.device)
                }

        # Get denormalized predictions for physics calculations
        if self.use_heteroscedastic:
            outputs_denorm = predictions_denorm
        else:
            outputs_denorm = outputs_denorm_for_mse
        
        # In OPF mode, feature 0 varies by bus type (V for PQ, Q for PV, P for Slack)
        # Only check for negative voltages on PQ buses (where V is predicted) - NO CLAMPING
        pq_mask = (bus_types == 0)  # [batch, buses] - PQ buses only
        if pq_mask.any():
            vm_pu = outputs_denorm[..., 0]
            vm_pq = vm_pu[pq_mask]  # Only check PQ buses
            negative_vm_count = (vm_pq < 0).sum().item()
            negative_vm_fraction = negative_vm_count / vm_pq.numel() if vm_pq.numel() > 0 else 0.0
            
            # Removed negative voltage warnings - model is still learning, these are expected during training
            # if not hasattr(self, '_batch_count'):
            #     self._batch_count = 0
            # self._batch_count += 1
            # if negative_vm_count > 0 and self._batch_count % 100 == 0:
            #     min_vm = vm_pq.min().item()
            #     max_vm = vm_pq.max().item()
            #     print(f"Batch {self._batch_count}: {negative_vm_count} negative voltage predictions on PQ buses "
            #           f"({negative_vm_fraction*100:.1f}%), VM range: [{min_vm:.4f}, {max_vm:.4f}]")
        
        if measurements_norm.dim() == 4:
            # Sequential model: use last timestep [batch, seq_len, buses, 10] -> [batch, buses, 10]
            measurements_norm = measurements_norm[:, -1, :, :]  # Take last timestep
        elif measurements_norm.dim() != 3:
            raise ValueError(
                f"measurements_norm must be 3D [batch, buses, 10] or 4D [batch, seq_len, buses, 10], "
                f"but got shape {measurements_norm.shape}"
            )
        measurements_denorm = self.normalizer.denormalize(measurements_norm)
        
        # Pass bus_types to power violation computation (critical for OPF mode)
        # Use squared=True for training (MSE), but we'll compute RMSE for display
        power_violation_mse_per_sample = self._compute_power_balance_violation(
            predicted_voltages=outputs_denorm,
            measured_power=measurements_denorm,
            ybus_batch=ybus_batch,
            bus_types=bus_types,  # OPF: bus type codes [0=PQ, 1=PV, 2=Slack]
            squared=True,  # MSE for training loss
            debug=False  # Set to True for debugging
        )
        # Compute RMSE for display (more interpretable than MSE)
        power_violation_rmse_per_sample = self._compute_power_balance_violation(
            predicted_voltages=outputs_denorm,
            measured_power=measurements_denorm,
            ybus_batch=ybus_batch,
            bus_types=bus_types,
            squared=False,  # RMSE for display
            debug=False
        )
        # Pass bus_types to voltage violation computation (only check PQ buses in OPF mode)
        # Compute both MSE (for training) and RMSE (for display)
        voltage_violation_mse_per_sample = self._compute_voltage_limit_violation(
            outputs_denorm, bus_types=bus_types, squared=True  # MSE for training
        )
        voltage_violation_rmse_per_sample = self._compute_voltage_limit_violation(
            outputs_denorm, bus_types=bus_types, squared=False  # RMSE for display
        )

        # Use MSE for training loss (squared, for optimization)
        power_penalty = torch.mean(power_violation_mse_per_sample)  # MSE (per-unit²) for training
        # Use RMSE for display (sqrt, more interpretable)
        power_penalty_rmse = torch.mean(power_violation_rmse_per_sample)  # RMSE (per-unit) for display
        voltage_penalty = torch.mean(voltage_violation_mse_per_sample)  # MSE (per-unit²) for training
        voltage_penalty_rmse = torch.mean(voltage_violation_rmse_per_sample)  # RMSE (per-unit) for display
        
        if self.use_heteroscedastic:
            # Heteroscedastic: Data loss uses natural parametrization (Immer et al., NeurIPS 2023)
            # Data loss: pure natural parametrization NLL (paper Equation 5)
            weighted_data_loss = data_loss  # Already computed as NLL
            
            # Physics losses: Use Kendall-style learnable weights (Kendall et al., CVPR 2018)
            # Heteroscedastic paper does NOT address physics losses - only data loss
            # Kendall-style weights are well-cited (2000+ citations) and appropriate for multi-task learning
            sigma_power = torch.exp(self.log_sigma_power)
            sigma_voltage = torch.exp(self.log_sigma_voltage)
            
            weighted_power_loss = (1.0 / (2.0 * sigma_power ** 2)) * power_penalty
            weighted_voltage_loss = (1.0 / (2.0 * sigma_voltage ** 2)) * voltage_penalty
            
            # Regularization: log(sigma) terms for physics losses (Kendall-style)
            # Data loss regularization is already in the NLL (natural parametrization)
            regularization = torch.log(sigma_power) + torch.log(sigma_voltage)
        else:
            # Homoscedastic: Use learnable parameters for all losses
            sigma_data = torch.exp(self.log_sigma_data)
            sigma_power = torch.exp(self.log_sigma_power)
            sigma_voltage = torch.exp(self.log_sigma_voltage)
            
            weighted_data_loss = (1.0 / (2.0 * sigma_data ** 2)) * data_loss
            weighted_power_loss = (1.0 / (2.0 * sigma_power ** 2)) * power_penalty
            weighted_voltage_loss = (1.0 / (2.0 * sigma_voltage ** 2)) * voltage_penalty
            
            regularization = torch.log(sigma_data) + torch.log(sigma_power) + torch.log(sigma_voltage)
        
        total_loss = weighted_data_loss + weighted_power_loss + weighted_voltage_loss + regularization
        
        # Compute denormalized MSE for reporting (physical units - what ML engineers typically show)
        # This is more interpretable than normalized MSE
        if self.use_heteroscedastic:
            mse_denorm = mse_var1_denorm + mse_var2_denorm  # Denormalized MSE (physical units)
            mse_normalized = unweighted_mse  # Normalized MSE (for internal consistency)
        else:
            # For homoscedastic, mse_var1_denorm and mse_var2_denorm were already computed above
            mse_denorm = mse_var1_denorm + mse_var2_denorm  # Denormalized MSE (physical units)
            # For homoscedastic, data_loss is already denormalized, so normalized = denormalized
            # But we normalize it for consistency with heteroscedastic case
            # Actually, for homoscedastic, we don't normalize, so normalized = denormalized
            mse_normalized = data_loss  # For homoscedastic, normalized = denormalized (no normalization applied)
        
        # Compute denormalized total_loss for display (physical units - interpretable)
        # Use RMSE for violations in display (more interpretable than MSE)
        # We keep normalized total_loss for optimization (backward pass, etc.)
        if self.is_physics_informed:
            # For display: use RMSE for violations (sqrt, per-unit) instead of MSE (squared, per-unit²)
            # This makes violations more interpretable and shows actual per-unit mismatch
            total_loss_denorm = mse_denorm + power_penalty_rmse + voltage_penalty  # Physical units (interpretable)
        else:
            total_loss_denorm = mse_denorm  # For non-physics models, just MSE
        
        # ML Engineering Best Practice: Report denormalized MSE and total_loss (physical units) for interpretability
        # Total loss remains normalized (for optimization stability)
        # Violations are already in physical units (per-unit)
        if return_components and self.is_physics_informed:
            result = {
                'total_loss': total_loss,  # Normalized (for optimization - backward pass, etc.)
                'total_loss_denorm': total_loss_denorm,  # Denormalized (physical units - for display)
                'mse': mse_denorm,  # Denormalized MSE (physical units - what ML engineers show)
                'mse_normalized': mse_normalized,  # Normalized MSE (for reference)
                'mse_var1': mse_var1_norm if self.use_heteroscedastic else mse_var1,
                'mse_var2': mse_var2_norm if self.use_heteroscedastic else mse_var2,
                'power_violation': power_penalty_rmse,  # RMSE (per-unit) for display - more interpretable
                'power_violation_mse': power_penalty,  # MSE (per-unit²) for training - kept for reference
                'voltage_violation': voltage_penalty_rmse,  # RMSE (per-unit) for display - more interpretable
                'voltage_violation_mse': voltage_penalty,  # MSE (per-unit²) for training - kept for reference
                'physics_loss': weighted_power_loss + weighted_voltage_loss
            }
            return result
        else:
            result = {
                'total_loss': total_loss,  # Normalized (for optimization - backward pass, etc.)
                'total_loss_denorm': total_loss_denorm,  # Denormalized (physical units - for display)
                'mse': mse_denorm,  # Denormalized MSE (physical units - what ML engineers show)
                'mse_normalized': mse_normalized,  # Normalized MSE (for reference)
                'mse_weighted': weighted_data_loss if not self.use_heteroscedastic else None,
                'mse_var1': mse_var1_norm if self.use_heteroscedastic else mse_var1,
                'mse_var2': mse_var2_norm if self.use_heteroscedastic else mse_var2,
                'power_violation': power_penalty_rmse,  # RMSE (per-unit) for display - more interpretable
                'power_violation_mse': power_penalty,  # MSE (per-unit²) for training - kept for reference
                'voltage_violation': voltage_penalty_rmse,  # RMSE (per-unit) for display - more interpretable
                'voltage_violation_mse': voltage_penalty  # MSE (per-unit²) for training - kept for reference
            }
            return result

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
    
    def _reconstruct_voltage_state(self, predicted_unknowns: torch.Tensor, measured_power: torch.Tensor, bus_types: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs full voltage state [V, θ] from bus-type-dependent predictions.
        
        CRITICAL: In OPF mode, predictions are bus-type dependent:
        - PQ bus (0): [V, θ] - both unknown, use predictions directly
        - PV bus (1): [Q, θ] - V is known (from measurements), Q and θ are predicted
        - Slack bus (2): [P, Q] - V and θ are both known (from measurements), P and Q are predicted
        
        Args:
            predicted_unknowns: Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
            measured_power: Measured power [batch, buses, 10] = [p_load, q_load, ..., vm_meas, va_meas]
            bus_types: Required [batch, buses] with codes [0=PQ, 1=PV, 2=Slack]
            
        Returns:
            Full voltage state [batch, buses, 2] = [vm_pu, va_rad] for ALL buses
        """
        if bus_types is None:
            raise ValueError("bus_types is required for OPF mode. This project uses OPF, not state estimation.")
        
        # Extract measured voltages from features (indices 8, 9)
        vm_meas_pu = measured_power[..., 8]  # [batch, buses] - measured voltage magnitude
        va_meas_rad = measured_power[..., 9]  # [batch, buses] - measured voltage angle
        
        # OPF mode: bus-type dependent predictions
        vm_pu = torch.zeros_like(vm_meas_pu)  # [batch, buses]
        va_rad = torch.zeros_like(va_meas_rad)  # [batch, buses]
        
        # Create masks for each bus type
        pq_mask = (bus_types == 0)  # [batch, buses] - PQ buses
        pv_mask = (bus_types == 1)  # [batch, buses] - PV buses
        slack_mask = (bus_types == 2)  # [batch, buses] - Slack buses
        
        # PQ buses: [V, θ] are both predicted (unknowns)
        vm_pu[pq_mask] = predicted_unknowns[..., 0][pq_mask]
        va_rad[pq_mask] = predicted_unknowns[..., 1][pq_mask]
        
        # PV buses: V is known (measured), θ is predicted
        vm_pu[pv_mask] = vm_meas_pu[pv_mask]  # Use measured V
        va_rad[pv_mask] = predicted_unknowns[..., 1][pv_mask]  # Use predicted θ
        
        # Slack buses: V and θ are both known (measured)
        vm_pu[slack_mask] = vm_meas_pu[slack_mask]  # Use measured V
        va_rad[slack_mask] = va_meas_rad[slack_mask]  # Use measured θ
        
        # Return full voltage state [batch, buses, 2] = [vm_pu, va_rad]
        return torch.stack([vm_pu, va_rad], dim=-1)
    
    def _compute_power_balance_violation(self, predicted_voltages, measured_power, ybus_batch, bus_types: torch.Tensor, squared=True, debug=False):
        """
        Computes power balance violation for OPF predictions.
        
        CRITICAL: In OPF mode, predictions are bus-type dependent:
        - PQ bus (0): [V, θ] - both unknown, use predictions directly
        - PV bus (1): [Q, θ] - V is known (from measurements), Q and θ are predicted
        - Slack bus (2): [P, Q] - V and θ are known (from measurements), P and Q are predicted
        
        To compute power flow, we need full voltage state [V, θ] for ALL buses.
        We reconstruct it by combining predictions with known measurements.
        
        Args:
            predicted_voltages: Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
            measured_power: Measured power injections [batch, buses, 10] = [p_load, q_load, ..., vm_meas, va_meas]
            ybus_batch: Admittance matrices [batch, buses, buses]
            bus_types: Required [batch, buses] with codes [0=PQ, 1=PV, 2=Slack]
            squared: If True, returns MSE (for training), if False, returns RMSE (for evaluation)
            debug: If True, print debug information
            
        Returns:
            Power balance violation per sample [batch]
        """
        if bus_types is None:
            raise ValueError("bus_types is required for OPF mode. This project uses OPF, not state estimation.")
        
        # Reconstruct full voltage state [V, θ] for ALL buses based on bus type
        voltage_state = self._reconstruct_voltage_state(predicted_voltages, measured_power, bus_types)
        vm_pu = voltage_state[..., 0]  # [batch, buses]
        va_rad = voltage_state[..., 1]  # [batch, buses]
        
        if debug:
            pq_mask = (bus_types == 0)
            pv_mask = (bus_types == 1)
            slack_mask = (bus_types == 2)
            print(f"[DEBUG] Bus type distribution: PQ={pq_mask.sum().item()}, PV={pv_mask.sum().item()}, Slack={slack_mask.sum().item()}")
            print(f"[DEBUG] VM range: PQ={vm_pu[pq_mask].min().item():.4f} to {vm_pu[pq_mask].max().item():.4f}")
            print(f"[DEBUG] VM range: PV={vm_pu[pv_mask].min().item():.4f} to {vm_pu[pv_mask].max().item():.4f}")
            print(f"[DEBUG] VM range: Slack={vm_pu[slack_mask].min().item():.4f} to {vm_pu[slack_mask].max().item():.4f}")
        
        # Compute power flow using reconstructed voltage state
        V = vm_pu * torch.exp(1j * va_rad)  # [batch, buses] - complex voltage
        I = torch.einsum('bij,bj->bi', ybus_batch.cfloat(), V)  # [batch, buses] - current
        S_calc_pu = V * torch.conj(I)  # [batch, buses] - calculated power injection
        
        # Get actual power injections from measurements
        p_inj_pu, q_inj_pu = self._get_power_injections_pu(measured_power)  # [batch, buses]
        
        # Compute mismatch
        p_mismatch = p_inj_pu - S_calc_pu.real  # [batch, buses]
        q_mismatch = q_inj_pu - S_calc_pu.imag  # [batch, buses]
        
        mismatch_squared = p_mismatch**2 + q_mismatch**2  # [batch, buses]
        
        if debug:
            print(f"[DEBUG] P mismatch range: {p_mismatch.min().item():.6f} to {p_mismatch.max().item():.6f}")
            print(f"[DEBUG] Q mismatch range: {q_mismatch.min().item():.6f} to {q_mismatch.max().item():.6f}")
            print(f"[DEBUG] Mismatch squared mean: {torch.mean(mismatch_squared).item():.6f}")
            print(f"[DEBUG] P injection range: {p_inj_pu.min().item():.6f} to {p_inj_pu.max().item():.6f}")
            print(f"[DEBUG] Q injection range: {q_inj_pu.min().item():.6f} to {q_inj_pu.max().item():.6f}")
            print(f"[DEBUG] S_calc real range: {S_calc_pu.real.min().item():.6f} to {S_calc_pu.real.max().item():.6f}")
            print(f"[DEBUG] S_calc imag range: {S_calc_pu.imag.min().item():.6f} to {S_calc_pu.imag.max().item():.6f}")
        
        if squared:
            return torch.mean(mismatch_squared, dim=-1)  # MSE for training [batch]
        else:
            return torch.sqrt(torch.mean(mismatch_squared, dim=-1))  # RMSE for evaluation [batch]

    def _compute_voltage_limit_violation(self, state: torch.Tensor, bus_types: torch.Tensor, squared=True) -> torch.Tensor:
        """
        Calculates the violation of voltage limits.
        
        CRITICAL: In OPF mode, state[..., 0] is bus-type dependent:
        - PQ bus: V (voltage magnitude) - check limits
        - PV bus: Q (reactive power) - NO voltage limit check (V is known/specified)
        - Slack bus: P (active power) - NO voltage limit check (V is known/specified)
        
        Args:
            state: Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
            bus_types: Required [batch, buses] with codes [0=PQ, 1=PV, 2=Slack]
            squared: If True, returns MSE (per-unit²) for training, if False, returns RMSE (per-unit) for display
            
        Returns:
            Voltage limit violation per sample [batch]
        """
        if bus_types is None:
            raise ValueError("bus_types is required for OPF mode. This project uses OPF, not state estimation.")
        
        # OPF mode: only check voltage limits for PQ buses (where V is predicted)
        batch_size = state.shape[0]
        vm_pu = state[..., 0]  # [batch, buses]
        
        # Compute violations for all buses (will be zero for non-PQ buses)
        v_below = F.relu(self.v_min - vm_pu)  # [batch, buses]
        v_above = F.relu(vm_pu - self.v_max)  # [batch, buses]
        violation_all = v_below**2 + v_above**2  # [batch, buses] - MSE (per-unit²)
        
        # Mask out non-PQ buses (set their violations to zero)
        pq_mask = (bus_types == 0)  # [batch, buses] - PQ buses
        violation_all = violation_all * pq_mask.float()  # Zero out non-PQ buses
        
        if squared:
            # Return MSE (per-unit²) for training
            return torch.mean(violation_all, dim=-1)  # [batch]
        else:
            # Return RMSE (per-unit) for display
            return torch.sqrt(torch.mean(violation_all, dim=-1))  # [batch]

    def _compute_normalized_active_power_loss(self, voltages: torch.Tensor, measurements: torch.Tensor, Ybus: torch.Tensor, bus_types: torch.Tensor) -> torch.Tensor:
        """
        Computes normalized active power loss using the accurate power loss formula from equation (3.5):
        P_loss = Σ Σ Dij * [Rij/|Vit||Vjt| * (PitPjt + QitQjt) + Rij|Vit||Vjt|sin(θit-θjt)(QitPjt - QjtPit)]
        
        Vectorized implementation for better performance.
        Normalized by total system load to ensure values are in [0, 1] range across all bus systems.
        
        CRITICAL: In OPF mode, voltages are bus-type dependent. This function reconstructs full voltage state.
        
        Args:
            voltages: Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
            measurements: Measured power [batch, buses, 10] = [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
            Ybus: Admittance matrix [batch, buses, buses]
            bus_types: Required [batch, buses] with codes [0=PQ, 1=PV, 2=Slack]
        """
        if bus_types is None:
            raise ValueError("bus_types is required for OPF mode. This project uses OPF, not state estimation.")
        # Reconstruct full voltage state [V, θ] for ALL buses
        voltage_state = self._reconstruct_voltage_state(voltages, measurements, bus_types)
        Vm = voltage_state[..., 0]  # Voltage magnitudes (p.u.) [batch_size, num_buses]
        Va = voltage_state[..., 1]  # Voltage angles (rad) [batch_size, num_buses]
        
        batch_size, num_buses = Vm.shape[:2]
        
        # Check if we have any data
        if batch_size == 0 or num_buses == 0:
            return torch.zeros(batch_size, device=Vm.device, dtype=Vm.dtype)
        
        # Calculate power losses from PREDICTED voltage state (model-dependent)
        # Use power flow equations to calculate losses from predicted voltages
        # Compute complex voltages: V = |V| * e^(jθ)
        V = Vm * torch.exp(1j * Va)  # [batch_size, num_buses] - complex voltage
        
        # Calculate currents using Ybus: I = Ybus * V
        I = torch.einsum('bij,bj->bi', Ybus.cfloat(), V)  # [batch_size, num_buses] - current
        
        # Calculate complex power injections: S = V * conj(I)
        S_inj = V * torch.conj(I)  # [batch_size, num_buses] - power injection
        p_inj = S_inj.real  # [batch_size, num_buses] - real power injection
        
        # Calculate power loss from predicted state using power balance: P_loss = P_gen - P_load
        # Get load from measurements (load is known, not predicted)
        p_load_mw = measurements[..., 0]  # Load (MW) - known from measurements
        p_load_total_pu = torch.sum(p_load_mw / self.s_base_mva, dim=-1)  # [batch_size]
        
        # Calculate generation from predicted state: P_gen = sum of positive power injections
        p_gen_pred_pu = torch.sum(F.relu(p_inj), dim=-1)  # [batch_size] - generation from predicted state
        
        # Power loss = generation - load (from power balance)
        p_losses_pu = p_gen_pred_pu - p_load_total_pu  # [batch_size]
        
        # Normalize by total load (from measurements) to get [0, 1] range
        epsilon = 1e-6  # Increased epsilon for stability
        total_load_pu = torch.abs(p_load_total_pu)  # [batch_size] - total load in per-unit (always positive)
        
        # Clamp total_load_pu to avoid division by very small numbers
        # Minimum load threshold: 0.01 p.u. (1% of base power) - prevents explosion
        min_load_threshold = 0.01
        total_load_pu_clamped = torch.clamp(total_load_pu, min=min_load_threshold)
        
        # Normalized loss = |losses| / (clamped_load + epsilon) - should be in [0, 1] range
        # Typical power losses are 1-5% of load, so values > 0.1 (10%) are unrealistic
        # Clamp normalized loss to reasonable range [0, 10] to prevent explosion from bad predictions
        normalized_loss = torch.abs(p_losses_pu) / (total_load_pu_clamped + epsilon)  # [batch_size]
        normalized_loss = torch.clamp(normalized_loss, min=0.0, max=10.0)  # Cap at 10x load (unrealistic but prevents NaN)
        
        return normalized_loss

    def _compute_normalized_voltage_deviation(self, voltages: torch.Tensor, measurements: torch.Tensor, bus_types: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized voltage deviation according to formula (3.6):
        f2 = Σ_t Σ_i |Vit - ViNt|/|ViNt|
        
        CRITICAL: In OPF mode, voltages are bus-type dependent. This function reconstructs full voltage state.
        Only checks PQ buses (where V is predicted).
        
        Args:
            voltages: Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
            measurements: Measured power [batch, buses, 10] (required for OPF mode to reconstruct voltage state)
            bus_types: Required [batch, buses] with codes [0=PQ, 1=PV, 2=Slack]
            
        Returns:
            Tensor containing normalized voltage deviations [batch_size]
        """
        if bus_types is None:
            raise ValueError("bus_types is required for OPF mode. This project uses OPF, not state estimation.")
        if measurements is None:
            raise ValueError("measurements is required for OPF mode to reconstruct voltage state.")
        
        # Reconstruct full voltage state [V, θ] for ALL buses
        voltage_state = self._reconstruct_voltage_state(voltages, measurements, bus_types)
        Vm = voltage_state[..., 0]  # [batch_size, num_buses]
        # Only check PQ buses (where V is predicted)
        pq_mask = (bus_types == 0)  # [batch, buses]
        Vm = Vm * pq_mask.float()  # Zero out non-PQ buses
        
        # Rated voltage is 1.0 p.u.
        V_rated = torch.ones_like(Vm)
        
        # Calculate absolute normalized deviation: |Vit - ViNt|/|ViNt|
        voltage_deviation = torch.abs(Vm - V_rated) / V_rated
        
        # Take mean across buses for each sample
        mean_deviation = torch.mean(voltage_deviation, dim=1)
        
        return mean_deviation

    def _compute_normalized_power_flow(self, voltages: torch.Tensor, measurements: torch.Tensor, Ybus: torch.Tensor, bus_types: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """
        Computes normalized power flow magnitudes using ACOPF equation (3.8):
        P_i^DG + P_i = P_i^load + V_i Σ_(s=1)^(B_n) V_s (G_is cos θ_is + B_is sin θ_is)
        Q_i^DG + Q_i = Q_i^load + V_i Σ_(s=1)^(B_n) V_s (G_is sin θ_is - B_is cos θ_is)
        
        This implementation calculates the actual power flow magnitudes through the network
        using the standard AC power flow equations, same foundation as power balance violation
        but measuring flow magnitudes instead of balance mismatches.
        
        CRITICAL: In OPF mode, voltages are bus-type dependent. This function reconstructs full voltage state.
        
        Args:
            voltages: Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent)
            measurements: Measured power [batch, buses, 10] = [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
            Ybus: Admittance matrix [batch_size, num_buses, num_buses]
            bus_types: Required [batch, buses] with codes [0=PQ, 1=PV, 2=Slack]
            epsilon: Small value to avoid division by zero
            
        Returns:
            Tensor containing normalized power flow magnitudes [batch_size]
        """
        if bus_types is None:
            raise ValueError("bus_types is required for OPF mode. This project uses OPF, not state estimation.")
        # Reconstruct full voltage state [V, θ] for ALL buses
        voltage_state = self._reconstruct_voltage_state(voltages, measurements, bus_types)
        vm_pu = voltage_state[..., 0]  # Voltage magnitudes (p.u.) [batch_size, num_buses]
        va_rad = voltage_state[..., 1]  # Voltage angles (rad) [batch_size, num_buses]
        
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
        
        # Clamp total_load_pu to avoid division by very small numbers
        min_load_threshold = 0.01
        total_load_pu_clamped = torch.clamp(torch.abs(total_load_pu), min=min_load_threshold)
        
        normalized_power_flow = mean_flow_magnitude_per_bus / (total_load_pu_clamped + epsilon)
        normalized_power_flow = torch.clamp(normalized_power_flow, min=0.0, max=100.0)  # Cap at 100x load
        
        return normalized_power_flow

    def _compute_carbon_emissions(
        self, 
        measurements: torch.Tensor, 
        time_carbon_coeff: torch.Tensor, 
        time_energy_coeff: torch.Tensor,
        renewable_fraction: torch.Tensor = None,
        voltages: torch.Tensor = None,
        Ybus: torch.Tensor = None,
        bus_types: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes carbon emissions using PREDICTED state (model-dependent).
        
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
            voltages: REQUIRED - Predicted unknowns [batch, buses, 2] (OPF: bus-type dependent) - used to calculate generation from predicted state
            Ybus: REQUIRED - Admittance matrix [batch, buses, buses] - used to calculate generation from predicted state
            bus_types: REQUIRED - [batch, buses] with codes [0=PQ, 1=PV, 2=Slack] - used to reconstruct voltage state
        """
        # Require predictions - no fallback to hide problems
        if voltages is None or Ybus is None or bus_types is None:
            raise ValueError(
                "Carbon emissions calculation requires predictions (voltages, Ybus, bus_types). "
                "No fallback to measurements - this ensures model-dependent evaluation."
            )
        
        # Reconstruct full voltage state from predictions
        voltage_state = self._reconstruct_voltage_state(voltages, measurements, bus_types)
        Vm = voltage_state[..., 0]  # [batch_size, num_buses]
        Va = voltage_state[..., 1]  # [batch_size, num_buses]
        
        # Calculate complex voltages: V = |V| * e^(jθ)
        V = Vm * torch.exp(1j * Va)  # [batch_size, num_buses]
        
        # Calculate currents: I = Ybus * V
        I = torch.einsum('bij,bj->bi', Ybus.cfloat(), V)  # [batch_size, num_buses]
        
        # Calculate power injections: S = V * conj(I)
        S_inj = V * torch.conj(I)  # [batch_size, num_buses]
        p_inj = S_inj.real  # [batch_size, num_buses] - real power injection
        
        # Total generation from predicted state (positive injections)
        p_gen_total_pred = torch.sum(F.relu(p_inj), dim=-1)  # [batch_size] - generation in p.u.
        p_gen_total_pred_mw = p_gen_total_pred * self.s_base_mva  # [batch_size] - generation in MW
        
        # Get renewable fraction from measurements (to determine conventional vs renewable split)
        # Format: [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
        p_ren_mw_meas = measurements[..., 6]  # Renewable generation from measurements (MW)
        p_conv_mw_meas = measurements[..., 4]  # Conventional generation from measurements (MW)
        total_ren_meas = torch.sum(p_ren_mw_meas, dim=-1)  # [batch_size]
        total_conv_meas = torch.sum(p_conv_mw_meas, dim=-1)  # [batch_size]
        total_gen_meas = total_ren_meas + total_conv_meas  # [batch_size]
        
        # Calculate renewable fraction from measurements
        epsilon = 1e-9
        ren_frac = total_ren_meas / (total_gen_meas + epsilon)  # [batch_size]
        
        # Apply renewable fraction to predicted total generation
        total_ren_pred_mw = p_gen_total_pred_mw * ren_frac  # [batch_size]
        total_conv_pred_mw = p_gen_total_pred_mw * (1.0 - ren_frac)  # [batch_size]
        
        # External grid: use predicted state to determine import/export
        p_load_mw = measurements[..., 0]  # Load (MW) - known from measurements
        p_load_total = torch.sum(p_load_mw, dim=-1)  # [batch_size]
        
        # Net power needed = load - renewable generation
        p_net_needed = p_load_total - total_ren_pred_mw  # [batch_size]
        
        # External grid import = max(0, net_power_needed - conventional_generation)
        # If conventional generation is insufficient, import from grid
        p_ext_import_pred = F.relu(p_net_needed - total_conv_pred_mw)  # [batch_size]
        
        total_carbon_emitting_gen = total_conv_pred_mw + p_ext_import_pred  # [batch_size]
        total_renewable_gen = total_ren_pred_mw  # [batch_size]
        
        carbon_intensity = time_carbon_coeff.squeeze(-1) if time_carbon_coeff.dim() > 1 else time_carbon_coeff
        energy_coefficient = time_energy_coeff.squeeze(-1) if time_energy_coeff.dim() > 1 else time_energy_coeff
        
        raw_emissions = (total_carbon_emitting_gen * carbon_intensity) / (energy_coefficient + 1e-9)
        total_generation = total_carbon_emitting_gen + total_renewable_gen
        
        # Clamp total_generation to avoid division by very small numbers
        min_gen_threshold = 0.01 * self.s_base_mva  # 1% of base power in MW
        total_generation_clamped = torch.clamp(total_generation, min=min_gen_threshold)
        
        normalized_emissions = total_carbon_emitting_gen / (total_generation_clamped + epsilon)
        # Clamp to [0, 1] range (should already be in this range, but ensure it)
        normalized_emissions = torch.clamp(normalized_emissions, min=0.0, max=1.0)
        
        return {'raw': raw_emissions, 'normalized': normalized_emissions}