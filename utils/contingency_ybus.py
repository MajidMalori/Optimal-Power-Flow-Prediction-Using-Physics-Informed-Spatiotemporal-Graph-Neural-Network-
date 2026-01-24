"""
Utilities for modifying Ybus matrices to simulate line outages (N-1 contingencies).
"""

# Custom exception for severe/unrecoverable errors that should stop data generation
# Defined FIRST to avoid import issues
class DataGenerationError(Exception):
    """
    Severe error during data generation that indicates fundamental problems.
    These errors should STOP execution - the data cannot be trusted.
    
    Examples:
    - Zero impedance (network data corruption)
    - Reactive power limit violations (control logic error)
    - Other fundamental physics violations
    """
    pass

import numpy as np
import torch
import warnings

try:
    import pandapower as pp
    PANDAPOWER_AVAILABLE = True
except ImportError:
    PANDAPOWER_AVAILABLE = False


def modify_ybus_for_line_outage(ybus: np.ndarray, net, line_idx: int) -> np.ndarray:
    """
    Modify Ybus matrix to simulate a line outage (N-1 contingency).
    
    This function removes the admittance contribution of the specified line
    from the Ybus matrix by subtracting its series and shunt admittances.
    
    Args:
        ybus: Base Ybus matrix [num_buses, num_buses] (complex)
        net: pandapower network (to get line parameters)
        line_idx: Index of line to remove
        
    Returns:
        Modified Ybus matrix with line removed [num_buses, num_buses] (complex)
    """
    if not PANDAPOWER_AVAILABLE:
        raise ImportError("pandapower is required for contingency Ybus modification")
    
    # Create a copy to avoid modifying the original
    ybus_modified = ybus.copy()
    
    # Check if line exists and is in service
    if line_idx not in net.line.index:
        warnings.warn(f"Line {line_idx} does not exist in network. Returning original Ybus.")
        return ybus_modified
    
    line = net.line.loc[line_idx]
    if not line.in_service:
        warnings.warn(f"Line {line_idx} is already out of service. Returning original Ybus.")
        return ybus_modified
    
    from_bus = int(line.from_bus)
    to_bus = int(line.to_bus)
    
    # Calculate line admittance (same as in calculate_ybus_from_net)
    r_ohm = line.r_ohm_per_km * line.length_km
    x_ohm = line.x_ohm_per_km * line.length_km
    z_series = r_ohm + 1j * x_ohm
    
    # NO FALLBACK: Zero impedance is unphysical and indicates data corruption
    if abs(z_series) < 1e-10:  # Check for near-zero (accounting for floating point)
        raise DataGenerationError(
            f"SEVERE ERROR: Line {line_idx} has zero or near-zero impedance: "
            f"r={r_ohm:.6e} ohm, x={x_ohm:.6e} ohm, z={z_series:.6e} ohm. "
            f"This is unphysical and indicates network data corruption. "
            f"Data generation STOPPED - cannot generate valid data with corrupted network."
        )
    
    y_series = 1.0 / z_series
    
    # Shunt admittance (line charging) - half at each end
    b_shunt_siemens = 1j * line.c_nf_per_km * line.length_km * 2 * np.pi * net.f_hz * 1e-9
    y_shunt_half = b_shunt_siemens / 2.0
    
    # Remove line admittance from Ybus (reverse of addition in calculate_ybus_from_net)
    # Remove from off-diagonal elements
    ybus_modified[from_bus, to_bus] += y_series  # Add back (subtract negative)
    ybus_modified[to_bus, from_bus] += y_series
    
    # Remove from diagonal elements
    ybus_modified[from_bus, from_bus] -= (y_series + y_shunt_half)
    ybus_modified[to_bus, to_bus] -= (y_series + y_shunt_half)
    
    return ybus_modified


def modify_adjacency_for_line_outage(adjacency: np.ndarray, net, line_idx: int) -> np.ndarray:
    """
    Modify adjacency matrix to simulate a line outage (N-1 contingency).
    
    This function removes the connection between buses for the specified line.
    
    Args:
        adjacency: Base adjacency matrix [num_buses, num_buses] (real)
        net: pandapower network (to get line connections)
        line_idx: Index of line to remove
        
    Returns:
        Modified adjacency matrix with line removed [num_buses, num_buses] (real)
    """
    if not PANDAPOWER_AVAILABLE:
        raise ImportError("pandapower is required for contingency adjacency modification")
    
    # Create a copy to avoid modifying the original
    adj_modified = adjacency.copy()
    
    # Check if line exists and is in service
    if line_idx not in net.line.index:
        warnings.warn(f"Line {line_idx} does not exist in network. Returning original adjacency.")
        return adj_modified
    
    line = net.line.loc[line_idx]
    if not line.in_service:
        warnings.warn(f"Line {line_idx} is already out of service. Returning original adjacency.")
        return adj_modified
    
    from_bus = int(line.from_bus)
    to_bus = int(line.to_bus)
    
    # Remove connection (set to 0)
    adj_modified[from_bus, to_bus] = 0.0
    adj_modified[to_bus, from_bus] = 0.0
    
    return adj_modified


def create_contingency_ybus_batch(ybus_batch: torch.Tensor, net, line_idx: int) -> torch.Tensor:
    """
    Create a batch of modified Ybus matrices for a line outage contingency.
    
    Args:
        ybus_batch: Batch of Ybus matrices [batch_size, num_buses, num_buses] (complex)
        net: pandapower network
        line_idx: Index of line to remove
        
    Returns:
        Modified Ybus batch [batch_size, num_buses, num_buses] (complex)
    """
    # Convert to numpy for modification
    ybus_np = ybus_batch.cpu().numpy()
    
    # Get base Ybus (first element, assuming all are the same)
    ybus_base = ybus_np[0]
    
    # Modify base Ybus
    ybus_modified_base = modify_ybus_for_line_outage(ybus_base, net, line_idx)
    
    # Create batch by repeating modified Ybus
    batch_size = ybus_np.shape[0]
    ybus_modified_batch = np.stack([ybus_modified_base] * batch_size, axis=0)
    
    # Convert back to torch tensor
    return torch.from_numpy(ybus_modified_batch).to(ybus_batch.device).to(ybus_batch.dtype)


def create_contingency_adjacency_batch(adjacency_batch: torch.Tensor, net, line_idx: int) -> torch.Tensor:
    """
    Create a batch of modified adjacency matrices for a line outage contingency.
    
    Args:
        adjacency_batch: Batch of adjacency matrices [batch_size, num_buses, num_buses] (real)
        net: pandapower network
        line_idx: Index of line to remove
        
    Returns:
        Modified adjacency batch [batch_size, num_buses, num_buses] (real)
    """
    # Convert to numpy for modification
    adj_np = adjacency_batch.cpu().numpy()
    
    # Get base adjacency (first element, assuming all are the same)
    adj_base = adj_np[0]
    
    # Modify base adjacency
    adj_modified_base = modify_adjacency_for_line_outage(adj_base, net, line_idx)
    
    # Create batch by repeating modified adjacency
    batch_size = adj_np.shape[0]
    adj_modified_batch = np.stack([adj_modified_base] * batch_size, axis=0)
    
    # Convert back to torch tensor
    return torch.from_numpy(adj_modified_batch).to(adjacency_batch.device).to(adjacency_batch.dtype)


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Normalize adjacency matrix using the Renormalization Trick (Kipf & Welling).
    D^(-0.5) * (A + I) * D^(-0.5)
    
    Args:
        adj: Adjacency matrix [batch_size, num_nodes, num_nodes] or [num_nodes, num_nodes]
        
    Returns:
        Normalized adjacency matrix
    """
    # Handle both batched and unbatched adjacency matrices
    if adj.dim() == 2:
        # Unbatched: [num_nodes, num_nodes] -> [1, num_nodes, num_nodes]
        adj = adj.unsqueeze(0)
        was_unbatched = True
    else:
        was_unbatched = False
        
    # 1. Add self-loops: A_hat = A + I
    num_nodes = adj.shape[-1]
    identity = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0)
    adj_hat = adj + identity
    
    # 2. Calculate degree matrix D_hat
    # Degree is sum of rows (or cols since symmetric)
    degree = torch.sum(adj_hat, dim=2)  # [batch_size, num_nodes]
    
    # 3. Calculate D_hat^(-0.5)
    # Add epsilon to prevent division by zero
    epsilon = 1e-8
    degree = degree + epsilon
    degree_inv_sqrt = torch.pow(degree, -0.5)
    
    # Clip for numerical stability (prevent Inf)
    degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)
    
    # Create diagonal matrix from vector
    # degree_matrix_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # [batch_size, num_nodes, num_nodes]
    
    # 4. Symmetric normalization: D^(-0.5) * A_hat * D^(-0.5)
    # OPTIMIZED: Use broadcasting instead of matrix multiplication
    # D^(-0.5) * A * D^(-0.5) is equivalent to element-wise multiplication: A_ij * d_i * d_j
    # shape: [batch, nodes, 1] * [batch, nodes, nodes] * [batch, 1, nodes]
    degree_inv_sqrt_col = degree_inv_sqrt.unsqueeze(2) # [batch, nodes, 1]
    degree_inv_sqrt_row = degree_inv_sqrt.unsqueeze(1) # [batch, 1, nodes]
    
    adj_norm = adj_hat * degree_inv_sqrt_col * degree_inv_sqrt_row
    
    # Return in original shape
    if was_unbatched:
        return adj_norm.squeeze(0)
    
    return adj_norm

