import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

class PowerSystemNormalizer:
    """
    A class to handle normalization and de-normalization of power system features.
    
    Optimal Power Flow (OPF) Approach:
    - Features (inputs): 10 measurements [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
    - Targets (outputs): 2 unknowns per bus, bus-type dependent [PQ: V,θ | PV: Q,θ | Slack: P,Q]
    
    All targets are in consistent units for proper normalization:
    - V: per-unit (0.95-1.05)
    - θ: radians (-0.5 to 0.5)
    - P, Q: per-unit (converted from MW/MVar by dividing by S_BASE)
    This ensures the normalizer can compute meaningful mean/std across all bus types.
    """
    def __init__(self, features, targets):
        """
        Args:
            features: Input measurements [samples, buses, 10] (in MW/MVar, pu, rad)
            targets: Output unknowns [samples, buses, 2] (all in pu or radians for consistent scaling)
        """
        # Normalize features (measurements) - use float32 for memory efficiency
        self.feature_mean = np.mean(features, axis=(0, 1), dtype=np.float32).astype(np.float32)  # [10]
        self.feature_std = np.std(features, axis=(0, 1), dtype=np.float32).astype(np.float32)    # [10]
        self.feature_std[self.feature_std == 0] = 1.0
        
        # Normalize targets (voltages) - use float32 for memory efficiency
        self.target_mean = np.mean(targets, axis=(0, 1), dtype=np.float32).astype(np.float32)    # [2]
        self.target_std = np.std(targets, axis=(0, 1), dtype=np.float32).astype(np.float32)      # [2]
        self.target_std[self.target_std == 0] = 1.0
        
        # Legacy support: Use feature stats as default
        self.mean = self.feature_mean
        self.std = self.feature_std

    def normalize(self, data):
        """
        Normalize data using z-score normalization.
        Auto-detects whether data is features (10 dims) or targets (2 dims).
        
        Args:
            data: Input data [samples, buses, num_features] or tensor equivalent
            
        Returns:
            Normalized data in same format as input
        """
        # Handle both numpy arrays and PyTorch tensors
        if torch.is_tensor(data):
            # If it's a CUDA tensor, move to CPU first
            if data.is_cuda:
                data_cpu = data.detach().cpu()
            else:
                data_cpu = data.detach().cpu()
            # Convert to numpy for computation
            data_np = data_cpu.numpy()
            was_tensor = True
            original_device = data.device if data.is_cuda else None
        else:
            data_np = data
            was_tensor = False
            original_device = None
        
        # Auto-detect: features (10) or targets (2)?
        if data_np.ndim == 3:
            num_features = data_np.shape[-1]
        elif data_np.ndim == 2:
            # Flattened: try to infer from shape
            # This is ambiguous, but we'll try to match based on total elements
            num_features = data_np.shape[-1]
        else:
            # Fallback to feature stats
            num_features = len(self.feature_mean)
        
        if num_features == 10:
            # Features (measurements)
            mean_np = self.feature_mean
            std_np = self.feature_std
        elif num_features == 2:
            # Targets (unknowns)
            mean_np = self.target_mean
            std_np = self.target_std
        else:
            # Fallback: use feature stats (for partial features)
            mean_np = self.feature_mean[:num_features] if num_features <= len(self.feature_mean) else self.mean[:num_features]
            std_np = self.feature_std[:num_features] if num_features <= len(self.feature_std) else self.std[:num_features]
        
        # Ensure float32 for memory efficiency (reduce memory by 50% vs float64)
        data_np = data_np.astype(np.float32) if data_np.dtype != np.float32 else data_np
        mean_np = mean_np.astype(np.float32) if mean_np.dtype != np.float32 else mean_np
        std_np = std_np.astype(np.float32) if std_np.dtype != np.float32 else std_np
        
        result = (data_np - mean_np) / std_np
        result = result.astype(np.float32)  # Ensure float32 output
        
        # Convert back to tensor if input was a tensor
        if was_tensor:
            result_tensor = torch.from_numpy(result).float()
            if original_device is not None:
                result_tensor = result_tensor.to(original_device)
            return result_tensor
        else:
            return result

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data back to physical units.
        Auto-detects whether data is features (10) or targets (2).
        
        Args:
            data: Tensor of shape [batch_size, num_buses, num_features]
            
        Returns:
            Denormalized tensor of the same shape
        """
        # Handle both torch tensors and numpy arrays
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = torch.from_numpy(data).float()
        
        if data.dim() != 3:
            raise ValueError(
                f"denormalize expects a 3D tensor [batch_size, num_buses, num_features], "
                f"but got {data.dim()}D tensor with shape {data.shape}.\n"
                f"Please reshape your data before calling denormalize().\n"
                f"Example: data.view(batch_size, num_buses, num_features)"
            )
        
        num_features = data.shape[-1]
        
        # Auto-detect: features (10) or targets (2)?
        if num_features == 10:
            # Features (measurements)
            mean_np = self.feature_mean
            std_np = self.feature_std
        elif num_features == 2:
            # Targets (voltages)
            mean_np = self.target_mean
            std_np = self.target_std
        else:
            # Fallback: use slice (for partial features)
            mean_np = self.feature_mean[:num_features]
            std_np = self.feature_std[:num_features]
        
        # Convert to tensors on the same device as data
        mean_tensor = torch.from_numpy(mean_np).float().to(data.device)
        std_tensor = torch.from_numpy(std_np).float().to(data.device)
        
        # Simple denormalization: x_original = x_normalized * std + mean
        denormalized_data = data * std_tensor + mean_tensor
        
        # Return numpy if input was numpy, otherwise return tensor
        if is_numpy:
            return denormalized_data.numpy()
        return denormalized_data

    def denormalize_targets(self, data: torch.Tensor) -> torch.Tensor:
        """
        Explicitly denormalize targets (2 dimensions).
        Wrapper around denormalize for compatibility with plotting scripts.
        
        Args:
            data: Tensor of shape [batch_size, num_buses, 2]
            
        Returns:
            Denormalized tensor of the same shape
        """
        return self.denormalize(data)

class PowerSystemLazyDataset(Dataset):
    """
    Professional lazy-loading PyTorch Dataset for power system time-series data.
    
    This dataset only stores file paths and metadata, loading data on-demand in __getitem__.
    This is the scalable, memory-efficient approach for large datasets (e.g., 118-bus with 72K samples).
    
    The dataset never holds the full data in RAM - only the specific timestep(s) needed for each batch.
    
    PERFORMANCE FIX: Pre-normalized adjacency matrices are cached for all unique topologies.
    This eliminates redundant normalization operations during training.
    """
    def __init__(self, file_metadata, adjacency_matrix, normalizer, ybus_metadata, 
                 is_static, sequence_length=1, hours_per_day=24, topology_cache=None, topology_ids=None):
        """
        Args:
            file_metadata: List of dicts, each containing file paths and metadata for one sample
            adjacency_matrix: Base adjacency matrix [num_buses, num_buses] (for backward compatibility)
            normalizer: PowerSystemNormalizer instance (computed once from sample data)
            ybus_metadata: Dict with 'base_path', 'contingency_timesteps_path', 'contingency_matrices_path', 
                          and 'contingency_lookup' (dict mapping global timestep -> contingency index)
            is_static: Whether this is a static model (single timestep) or sequential
            sequence_length: Length of input sequence for sequential models
            hours_per_day: Number of hours per day (always 24 for time-series mode)
            topology_cache: Dict mapping topology_id -> pre-normalized adjacency tensor [num_buses, num_buses]
                          If None, falls back to single pre-normalized base adjacency
            topology_ids: Array mapping sample index -> topology_id (for lookup in topology_cache)
                         If None, all samples use topology_id=0 (base)
        """
        self.file_metadata = file_metadata  # List of dicts with file paths and indices
        self.adjacency_matrix = adjacency_matrix
        self.normalizer = normalizer
        self.ybus_metadata = ybus_metadata
        self.is_static = is_static
        self.sequence_length = sequence_length
        self.hours_per_day = hours_per_day
        
        # Load base Ybus matrix (REQUIRED - No fallback allowed)
        if ybus_metadata and 'base_path' in ybus_metadata and os.path.exists(ybus_metadata['base_path']):
            self.ybus_base = torch.from_numpy(np.load(ybus_metadata['base_path'], mmap_mode='r').copy()).cfloat()
        else:
            # Raise error immediately if base Ybus is missing
            raise RuntimeError(
                f"Ybus base matrix not found at {ybus_metadata.get('base_path') if ybus_metadata else 'None'}. "
                f"This is required for physics-informed training."
            )
            
        # Store topology cache
        self.topology_cache = topology_cache
        self.topology_ids = topology_ids

    def __len__(self):
        return len(self.file_metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        target_meta = self.file_metadata[idx]
        target_idx_in_file = target_meta['index_in_file']
        
        if self.is_static:
            # Load features for this timestep (copy to make writable for PyTorch)
            features = np.load(target_meta['features_path'], mmap_mode='r')[target_idx_in_file].copy()
            features_tensor = torch.from_numpy(features).float()
            
            # Load target for this timestep (copy to make writable)
            targets = np.load(target_meta['targets_path'], mmap_mode='r')[target_idx_in_file].copy()
            target_tensor = torch.from_numpy(targets).float()
            
            # Normalize on-the-fly
            features_tensor = self.normalizer.normalize(features_tensor.unsqueeze(0)).squeeze(0)
            target_tensor = self.normalizer.normalize(target_tensor.unsqueeze(0)).squeeze(0)
            
        else:
            # Sequential model: load a sequence of timesteps
            start_idx = idx
            end_idx = idx + self.sequence_length
            target_idx = end_idx
            
            # Load features for the sequence (copy to make writable)
            features_list = []
            for i in range(start_idx, end_idx):
                if i < len(self.file_metadata):
                    meta = self.file_metadata[i]
                    features = np.load(meta['features_path'], mmap_mode='r')[meta['index_in_file']].copy()
                    features_list.append(features)
            
            # Load target (the timestep immediately following the sequence)
            if target_idx < len(self.file_metadata):
                target_meta = self.file_metadata[target_idx]
                targets = np.load(target_meta['targets_path'], mmap_mode='r')[target_meta['index_in_file']].copy()
                target_tensor = torch.from_numpy(targets).float()
            else:
                # Fallback if target_idx is out of bounds
                raise IndexError(f"Target index {target_idx} out of bounds for dataset of size {len(self.file_metadata)}")
            
            # Stack features into sequence
            features_array = np.stack(features_list, axis=0)
            features_tensor = torch.from_numpy(features_array).float()
            
            # Normalize on-the-fly
            features_tensor = self.normalizer.normalize(features_tensor)
            target_tensor = self.normalizer.normalize(target_tensor.unsqueeze(0)).squeeze(0)
        
        # Get Ybus matrix - lazy load if contingency exists (copy to make writable)
        # Use per-file contingency info from metadata (not global lookup)
        if self.ybus_base is not None:
            # Check if this timestep has a contingency in its file
            file_contingency_path = target_meta.get('ybus_contingency_matrices_path', None)
            cont_local_idx = target_meta.get('ybus_contingency_local_idx', None)
            
            if file_contingency_path and cont_local_idx is not None and os.path.exists(file_contingency_path):
                # Load this file's contingency matrices
                contingency_matrices = np.load(file_contingency_path, mmap_mode='r')
                # Validate index is within bounds
                if 0 <= cont_local_idx < contingency_matrices.shape[0]:
                    ybus_for_item = torch.from_numpy(contingency_matrices[cont_local_idx].copy()).cfloat()
                else:
                    # Index out of bounds - this is a data corruption issue, raise error
                    raise IndexError(
                        f"Contingency index {cont_local_idx} out of bounds for file '{file_contingency_path}' "
                        f"(size: {contingency_matrices.shape[0]}). This indicates data corruption or mismatch. "
                        f"Global timestep: {target_meta.get('global_timestep', idx)}, "
                        f"File index: {target_meta.get('index_in_file', 'unknown')}"
                    )
            else:
                # Use base Ybus (no contingency for this timestep)
                ybus_for_item = self.ybus_base
        else:
            # NO FALLBACK: Ybus is required for physics-informed models
            raise RuntimeError(
                f"Ybus base matrix not available for sample {idx}. "
                f"Ybus metadata was not properly initialized during data loading. "
                f"This indicates a data loading error."
            )
        
        # Load coefficients and metadata
        energy_coeffs = np.loadtxt(target_meta['energy_path'])
        carbon_coeffs = np.loadtxt(target_meta['carbon_path'])
        # Handle both 1D and 2D coefficient arrays
        if energy_coeffs.ndim == 1:
            time_energy = energy_coeffs[target_meta['index_in_file']] if len(energy_coeffs) > target_meta['index_in_file'] else energy_coeffs[0]
        else:
            time_energy = energy_coeffs[target_meta['index_in_file']]
        
        if carbon_coeffs.ndim == 1:
            time_carbon = carbon_coeffs[target_meta['index_in_file']] if len(carbon_coeffs) > target_meta['index_in_file'] else carbon_coeffs[0]
        else:
            time_carbon = carbon_coeffs[target_meta['index_in_file']]
        
        renewable_fraction = target_meta['renewable_fraction']
        
        # Load bus types on-demand (copy to make writable)
        bus_types_tensor = None
        if target_meta.get('bus_types_path') and os.path.exists(target_meta['bus_types_path']):
            bus_types = np.load(target_meta['bus_types_path'], mmap_mode='r')
            if bus_types.ndim == 2:
                bus_types_tensor = torch.from_numpy(bus_types[target_meta['index_in_file']].copy()).long()
            else:
                bus_types_tensor = torch.from_numpy(bus_types.copy()).long()
        
        # OPF: Extract generation components from FEATURES (measurements)
        from config import FeatureIndices
        
        if features_tensor.dim() == 3:
            # Sequential model: use last timestep
            features_last = features_tensor[-1] if features_tensor.shape[0] > 1 else features_tensor[0]
        else:
            features_last = features_tensor
        
        ext_grid_gen = features_last[:, FeatureIndices.P_EXT_GRID:FeatureIndices.Q_EXT_GRID+1]
        conventional_gen = features_last[:, FeatureIndices.P_CONV:FeatureIndices.Q_CONV+1]
        renewable_gen = features_last[:, FeatureIndices.P_REN:FeatureIndices.Q_REN+1]
        
        # PERFORMANCE FIX: Get pre-normalized adjacency from cache
        # NO FALLBACKS: Topology cache is required for new format
        if self.topology_cache is None or self.topology_ids is None:
            raise RuntimeError(
                f"Topology cache not initialized. This indicates a data loading error. "
                f"Expected topology_cache and topology_ids to be set during data loading."
            )
        
        topology_id = self.topology_ids[idx]
        if topology_id not in self.topology_cache:
            raise KeyError(
                f"Topology ID {topology_id} not found in cache for sample {idx}. "
                f"Available topology IDs: {list(self.topology_cache.keys())}. "
                f"This indicates data corruption or mismatch between topology_ids and cache."
            )
        
        adjacency_for_item = self.topology_cache[topology_id]
        
        return {
            'features': features_tensor,
            'adjacency': adjacency_for_item,  # Pre-normalized adjacency (from cache or base)
            'ybus_matrix': ybus_for_item,
            'targets': target_tensor,
            'bus_types': bus_types_tensor,
            'time_energy_coeffs': torch.tensor(time_energy, dtype=torch.float32),
            'time_carbon_coeffs': torch.tensor(time_carbon, dtype=torch.float32),
            'renewable_fraction': torch.tensor(renewable_fraction, dtype=torch.float32),
            'ext_grid_gen': ext_grid_gen,
            'conventional_gen': conventional_gen,
            'renewable_gen': renewable_gen,
            'timestep': target_meta.get('global_timestep', idx)
        }

def _convert_edge_index_to_adj(edge_index, num_nodes):
    num_nodes = int(num_nodes)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    source_nodes = edge_index[0].astype(int)
    dest_nodes = edge_index[1].astype(int)
    adj[source_nodes, dest_nodes] = 1
    adj[dest_nodes, source_nodes] = 1
    return adj

def pre_normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Pre-compute the symmetrically normalized adjacency matrix with self-loops.
    
    This is a CRITICAL performance optimization: for static graph topologies (like power systems),
    the normalized adjacency matrix is constant and should be computed ONCE during data loading,
    not millions of times during training.
    
    Implements: D_hat^(-0.5) * A_hat * D_hat^(-0.5)
    Where: A_hat = A + I (adjacency with self-loops)
    
    Args:
        adj: Raw adjacency matrix [num_nodes, num_nodes] (numpy array)
    
    Returns:
        Normalized adjacency matrix [num_nodes, num_nodes] (numpy array, float32)
    """
    import torch
    
    # Convert to torch tensor for computation
    adj_tensor = torch.from_numpy(adj).float()
    num_nodes = adj_tensor.shape[0]
    
    # Step 1: Add self-loops (A_hat = A + I)
    identity = torch.eye(num_nodes, dtype=adj_tensor.dtype)
    adj_hat = adj_tensor + identity  # [num_nodes, num_nodes]
    
    # Step 2: Compute degree matrix D_hat
    degree = torch.sum(adj_hat, dim=1)  # [num_nodes] - degree of each node
    
    # Handle zero-degree nodes (isolated nodes) to avoid division by zero
    epsilon = 1e-8
    degree = degree + epsilon
    
    # Step 3: Symmetric normalization: D_hat^(-0.5) * A_hat * D_hat^(-0.5)
    degree_inv_sqrt = torch.pow(degree, -0.5)  # [num_nodes]
    degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)  # Prevent extreme values
    
    # Create diagonal matrix: D_hat^(-0.5)
    degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)  # [num_nodes, num_nodes]
    
    # Symmetric normalization: D_hat^(-0.5) @ A_hat @ D_hat^(-0.5)
    normalized_adj = degree_matrix_inv_sqrt @ adj_hat @ degree_matrix_inv_sqrt  # [num_nodes, num_nodes]
    
    # Convert back to numpy (float32 for memory efficiency)
    return normalized_adj.numpy().astype(np.float32)

def _build_topology_cache_from_ids(file_metadata, base_adjacency, num_buses, case_name, data_dir):
    """
    Build topology cache from topology_ids stored in metadata.
    
    This is the FULL performance optimization: pre-normalize all unique topologies
    (base + all contingencies) once during data loading, then use O(1) lookup during training.
    
    Args:
        file_metadata: List of metadata dicts with 'topology_id' for each sample
        base_adjacency: Base adjacency matrix [num_buses, num_buses]
        num_buses: Number of buses
        case_name: Case name (e.g., '33bus') for loading network
        data_dir: Data directory
    
    Returns:
        topology_cache: Dict mapping topology_id -> pre-normalized adjacency tensor
        topology_ids: Array mapping sample index -> topology_id
    """
    # Extract topology_ids from metadata
    topology_ids = np.array([meta.get('topology_id', 0) for meta in file_metadata], dtype=np.int32)
    unique_topology_ids = np.unique(topology_ids)
    
    base_count = np.sum(topology_ids == 0)
    num_contingencies = len(unique_topology_ids) - 1  # Exclude base (ID=0)
    # Building cache (silent - will print completion)
    
    # Pre-normalize base topology
    topology_cache = {}
    normalized_base = pre_normalize_adjacency(base_adjacency)
    topology_cache[0] = torch.from_numpy(normalized_base).float()
    
    # Pre-normalize all contingency topologies
    # NO FALLBACKS: All contingency topologies must be built correctly
    if len(unique_topology_ids) > 1:
        # Load pandapower network to modify adjacency for contingencies
        try:
            import pandapower.networks as pn
        except ImportError:
            raise ImportError(
                "pandapower is required to build contingency topologies. "
                "Install with: pip install pandapower"
            )
        
        # Load network for case
        if '33' in case_name:
            net = pn.case33bw()
        elif '57' in case_name:
            net = pn.case57()
        elif '118' in case_name:
            net = pn.case118()
        else:
            raise ValueError(
                f"Unknown case name: {case_name}. Supported cases: 33bus, 57bus, 118bus. "
                f"Cannot build contingency topologies for unknown case."
            )
        
        from utils.contingency_ybus import modify_adjacency_for_line_outage
        
        for topo_id in unique_topology_ids:
            if topo_id > 0:  # Skip base (already done)
                line_idx = topo_id - 1
                
                # Check if line exists - NO FALLBACK
                if line_idx not in net.line.index:
                    raise IndexError(
                        f"Line index {line_idx} (from topology_id {topo_id}) not found in network {case_name}. "
                        f"Valid line indices: {list(net.line.index)}. "
                        f"This indicates data corruption or mismatch between topology_ids and network."
                    )
                
                # Create contingency adjacency
                contingency_adj = modify_adjacency_for_line_outage(base_adjacency, net, line_idx)
                # Pre-normalize
                normalized_cont = pre_normalize_adjacency(contingency_adj)
                topology_cache[topo_id] = torch.from_numpy(normalized_cont).float()
    
    if num_contingencies > 0:
        print(f"done. Cache: {len(topology_cache)} topologies (base + {num_contingencies} contingencies)", flush=True)
    else:
        print(f"done. Cache: {len(topology_cache)} topology (base only)", flush=True)
    return topology_cache, topology_ids

def load_power_system_data(config, case_name):
    """
    Professional lazy-loading data loader.
    
    Creates a metadata manifest instead of loading all data into RAM.
    Only loads a sample of data for normalization statistics computation.
    The actual data is loaded on-demand in PowerSystemLazyDataset.__getitem__.
    """
    print(f"[Data] Creating lazy data manifest for {case_name}...", end=" ", flush=True)
    data_dir = getattr(config, 'DATA_DIR', './data')
    feature_files = sorted(glob.glob(os.path.join(data_dir, f"{case_name}_features_frac*.npy")))
    if not feature_files:
        raise FileNotFoundError(f"No data files found for pattern: '{case_name}_features_frac*.npy' in '{data_dir}'.")
    
    try:
        # Extract number of buses from case name
        num_buses = int(''.join(filter(str.isdigit, case_name)))
        
        # Load base adjacency matrix (new format: REQUIRED)
        # NO FALLBACKS: Old format is not supported. Data must be regenerated with new format.
        first_features_path = feature_files[0]
        base_adj_path = first_features_path.replace('features', 'base_adjacency')
        
        if not os.path.exists(base_adj_path):
            raise FileNotFoundError(
                f"REQUIRED: base_adjacency file not found at {base_adj_path}\n"
                f"The new topology caching system requires base_adjacency files.\n"
                f"Please regenerate your data using the updated data generation script:\n"
                f"  python data/main.py test <timesteps>\n"
                f"Old format (adjacency_array) is no longer supported."
            )
        
        # New format: base_adjacency file exists
        # Loading base adjacency matrix from new format (silent)
        adj_object_array = np.load(base_adj_path, allow_pickle=True)
        edge_index = adj_object_array[0]
        raw_base_adjacency = _convert_edge_index_to_adj(edge_index, num_buses)
        has_topology_ids = True
        
        if raw_base_adjacency.ndim != 2 or raw_base_adjacency.shape[0] != raw_base_adjacency.shape[1]:
            raise ValueError(f"Conversion to dense matrix failed. Final shape is not square: {raw_base_adjacency.shape}.")
        
        # Store raw base adjacency (will be used to build topology cache)
        base_adjacency_matrix = raw_base_adjacency
    except Exception as e:
        print(f"\nError: Failed during adjacency matrix loading and conversion: {e}")
        raise
    
    # Build file metadata manifest (lightweight - only stores paths and indices)
    file_metadata = []
    global_timestep = 0
    all_contingency_timesteps = []
    ybus_base_path = None
    ybus_contingency_timesteps_path = None
    ybus_contingency_matrices_path = None
    
    # Load one sample of each file type for normalization statistics (one-time memory hit)
    print("Pre-loading sample data for normalization...", end=" ", flush=True)
    all_features_for_norm = []
    all_targets_for_norm = []
    
    import re
    for f_path in feature_files:
        targets_path = f_path.replace('features', 'targets')
        bus_types_path = f_path.replace('features', 'bus_types')
        energy_path = f_path.replace('features', 'time_energy_coeffs').replace('.npy', '.txt')
        carbon_path = f_path.replace('features', 'time_carbon_coeffs').replace('.npy', '.txt')
        
        # Extract renewable fraction from filename
        frac_match = re.search(r'frac(\d+\.\d+)', os.path.basename(f_path))
        renewable_fraction = float(frac_match.group(1)) if frac_match else 0.0
        
        # Get file shape without loading (use mmap to read shape only)
        features_mmap = np.load(f_path, mmap_mode='r', allow_pickle=False)
        num_timesteps = features_mmap.shape[0]
        
        # Load a sample for normalization (first 1000 timesteps or all if smaller)
        sample_size = min(1000, num_timesteps)
        all_features_for_norm.append(features_mmap[:sample_size].astype(np.float32))
        targets_mmap = np.load(targets_path, mmap_mode='r', allow_pickle=False)
        all_targets_for_norm.append(targets_mmap[:sample_size].astype(np.float32))
        
        # Get Ybus metadata (first file sets the pattern)
        if ybus_base_path is None:
            ybus_base_path = f_path.replace('features', 'ybus_base')
            ybus_contingency_timesteps_path = f_path.replace('features', 'ybus_contingency_timesteps')
            ybus_contingency_matrices_path = f_path.replace('features', 'ybus_contingency_matrices')
        
        # Load contingency timesteps if they exist (for this file)
        file_contingency_timesteps_path = f_path.replace('features', 'ybus_contingency_timesteps')
        file_contingency_matrices_path = f_path.replace('features', 'ybus_contingency_matrices')
        
        # Load topology_ids (REQUIRED for new format)
        # NO FALLBACKS: topology_ids file is required
        file_topology_ids_path = f_path.replace('features', 'topology_ids')
        if not os.path.exists(file_topology_ids_path):
            raise FileNotFoundError(
                f"REQUIRED: topology_ids file not found at {file_topology_ids_path}\n"
                f"The new topology caching system requires topology_ids files.\n"
                f"Please regenerate your data using the updated data generation script:\n"
                f"  python data/main.py test <timesteps>"
            )
        
        file_topology_ids = np.load(file_topology_ids_path)
        if len(file_topology_ids) != num_timesteps:
            raise ValueError(
                f"topology_ids length ({len(file_topology_ids)}) doesn't match timesteps ({num_timesteps}) "
                f"for file {f_path}. This indicates data corruption or generation error."
            )
        
        # Track local contingency indices for this file
        file_contingency_local_indices = {}
        if os.path.exists(file_contingency_timesteps_path):
            contingency_timesteps = np.load(file_contingency_timesteps_path)
            # Map LOCAL timestep indices (within this file) to LOCAL contingency matrix indices
            for local_cont_idx, local_ts in enumerate(contingency_timesteps):
                # Store: local file timestep -> local contingency matrix index
                file_contingency_local_indices[int(local_ts)] = local_cont_idx
        
        # Create metadata entry for each timestep in this file
        for i in range(num_timesteps):
            # Get topology_id for this timestep (REQUIRED - no default)
            topology_id = int(file_topology_ids[i])
            
            entry = {
                'features_path': f_path,
                'targets_path': targets_path,
                'bus_types_path': bus_types_path if os.path.exists(bus_types_path) else None,
                'energy_path': energy_path,
                'carbon_path': carbon_path,
                'index_in_file': i,
                'global_timestep': global_timestep,
                'renewable_fraction': renewable_fraction,
                'topology_id': int(topology_id),  # Store topology_id in metadata
                # Store contingency info per file (not global)
                'ybus_contingency_matrices_path': file_contingency_matrices_path if os.path.exists(file_contingency_matrices_path) else None,
                'ybus_contingency_local_idx': file_contingency_local_indices.get(i, None)  # Local index in file's contingency matrices
            }
            file_metadata.append(entry)
            global_timestep += 1
    
    # Create normalizer from sample data (one-time memory hit)
    concatenated_features_sample = np.concatenate(all_features_for_norm, axis=0)
    concatenated_targets_sample = np.concatenate(all_targets_for_norm, axis=0)
    normalizer = PowerSystemNormalizer(concatenated_features_sample, concatenated_targets_sample)
    del all_features_for_norm, all_targets_for_norm  # Free memory immediately
    import gc
    # REMOVED: Aggressive GC slows down execution - only use if actually hitting OOM
    # gc.collect()
    print("done. Building manifest...", end=" ", flush=True)
    
    # Build Ybus metadata
    ybus_metadata = {}
    if ybus_base_path and os.path.exists(ybus_base_path):
        ybus_metadata['base_path'] = ybus_base_path
        # Note: contingency matrices are now stored per-file in metadata entries
        # No need for global contingency_lookup anymore
        ybus_metadata['contingency_lookup'] = {}  # Keep for backward compatibility, but not used
    else:
        # Fallback: no Ybus metadata (will use None in dataset)
        ybus_metadata = None
    
    print(f"done. Manifest: {len(file_metadata)} samples | Features: [samples, buses, 10] | Targets: [samples, buses, 2] | Normalization: on-demand", flush=True)
    
    # PERFORMANCE FIX: Build topology cache with all unique topologies
    # NO FALLBACKS: Topology cache is required for new format
    if not has_topology_ids:
        raise RuntimeError(
            "Topology IDs not detected. The new topology caching system requires base_adjacency files.\n"
            "Please regenerate your data using the updated data generation script."
        )
    
    print(f"[Data] Building topology cache...", end=" ", flush=True)
    topology_cache, topology_ids_array = _build_topology_cache_from_ids(
        file_metadata, base_adjacency_matrix, num_buses, case_name, data_dir
    )
    
    # Return metadata and lightweight objects (not full data arrays)
    # The dataset will be created in create_data_loaders
    return (file_metadata, base_adjacency_matrix, ybus_metadata, normalizer, topology_cache, topology_ids_array)

def _collate_static(batch):
    """
    Collate function for static models.
    Ensures adjacency matrix is consistently shaped as [batch_size, num_buses, num_buses].
    """
    # Extract adjacency matrix from first item (it's the same for all items)
    static_adj_matrix = batch[0]['adjacency']  # [num_buses, num_buses]
    batch_size = len(batch)
    
    # Expand adjacency to batch dimension: [batch_size, num_buses, num_buses]
    if static_adj_matrix.dim() == 2:
        static_adj_batch = static_adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
    elif static_adj_matrix.dim() == 3:
        # Already has batch dimension, but should be single matrix
        static_adj_batch = static_adj_matrix[0].unsqueeze(0).expand(batch_size, -1, -1)
    else:
        static_adj_batch = static_adj_matrix
    
    # Collate other items normally
    other_items_batch = [{k: v for k, v in item.items() if k != 'adjacency'} for item in batch]
    collated_batch = default_collate(other_items_batch)
    
    # Set the properly shaped adjacency matrix
    collated_batch['adjacency'] = static_adj_batch
    
    return collated_batch

def _collate_sequential_padded(batch):
    """
    Collate function for sequential models.
    Ensures adjacency matrix is consistently shaped as [batch_size, num_buses, num_buses].
    """
    # Extract adjacency matrix from first item (it's the same for all items)
    static_adj_matrix = batch[0]['adjacency']  # [num_buses, num_buses]
    batch_size = len(batch)
    
    # Expand adjacency to batch dimension: [batch_size, num_buses, num_buses]
    if static_adj_matrix.dim() == 2:
        static_adj_batch = static_adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
    elif static_adj_matrix.dim() == 3:
        # Already has batch dimension, but should be single matrix
        static_adj_batch = static_adj_matrix[0].unsqueeze(0).expand(batch_size, -1, -1)
    else:
        static_adj_batch = static_adj_matrix
    
    # Collate other items
    other_items_batch = [{k: v for k, v in item.items() if k not in ['features', 'targets', 'adjacency']} for item in batch]
    collated_batch = default_collate(other_items_batch)
    
    # Set the properly shaped adjacency matrix
    collated_batch['adjacency'] = static_adj_batch
    
    # Pad features sequences
    features_list = [item['features'] for item in batch]
    collated_batch['features'] = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    
    # Stack targets: [batch_size, num_buses, 2]
    collated_batch['targets'] = default_collate([item['targets'] for item in batch])
    
    return collated_batch

def create_data_loaders(file_metadata, adjacency, ybus_metadata, normalizer, config, is_static, 
                        topology_cache=None, topology_ids=None):
    """
    Create data loaders for power system OPF data using lazy loading.
    
    Args:
        file_metadata: List of dicts with file paths and metadata (from load_power_system_data)
        adjacency: Static adjacency matrix [num_buses, num_buses]
        ybus_metadata: Dict with Ybus file paths and contingency lookup
        normalizer: PowerSystemNormalizer instance
        config: Configuration object
        is_static: Whether this is a static model (single timestep) or sequential
        topology_cache: Dict mapping topology_id -> pre-normalized adjacency tensor (optional)
        topology_ids: Array mapping sample index -> topology_id (optional)
    """
    seq_len = 1 if is_static else getattr(config, 'SEQUENCE_LENGTH', 1)
    hours_per_day = getattr(config, 'HOURS_PER_DAY', 24)
    
    # Create lazy dataset
    dataset = PowerSystemLazyDataset(
        file_metadata, adjacency, normalizer, ybus_metadata,
        is_static, seq_len, hours_per_day=hours_per_day,
        topology_cache=topology_cache, topology_ids=topology_ids
    )
    dataset_size = len(dataset)
    
    # Extract renewable fractions from metadata for splitting
    renewable_fractions = np.array([meta['renewable_fraction'] for meta in file_metadata])
    
    # Get split mode from config
    split_mode = getattr(config, 'DATA_SPLIT_MODE', 'blocked_timeseries')
    if split_mode == 'stratified':
        split_mode = 'blocked_timeseries'
    
    if split_mode == 'blocked_timeseries':
        # BLOCKED TIME-SERIES SPLIT
        unique_fractions = np.unique(renewable_fractions)
        train_indices, val_indices, test_indices = [], [], []
        
        for frac in unique_fractions:
            frac_mask = renewable_fractions == frac
            frac_indices = np.where(frac_mask)[0]
            valid_frac_indices = [idx for idx in frac_indices if idx < dataset_size]
            
            if len(valid_frac_indices) == 0:
                continue
            
            n_frac = len(valid_frac_indices)
            n_train = int(config.TRAIN_SPLIT * n_frac)
            n_val = int(config.VAL_SPLIT * n_frac)
            n_test = n_frac - n_train - n_val
            
            frac_train_indices = valid_frac_indices[:n_train]
            frac_val_indices = valid_frac_indices[n_train:n_train + n_val]
            frac_test_indices = valid_frac_indices[n_train + n_val:]
            
            train_indices.extend(frac_train_indices)
            val_indices.extend(frac_val_indices)
            test_indices.extend(frac_test_indices)
        
        train_indices.sort()
        val_indices.sort()
        test_indices.sort()
        split_mode_str = "Blocked Time-Series"
        
    elif split_mode == 'chronological':
        # SIMPLE CHRONOLOGICAL SPLIT
        all_indices = list(range(dataset_size))
        n_train = int(config.TRAIN_SPLIT * dataset_size)
        n_val = int(config.VAL_SPLIT * dataset_size)
        n_test = dataset_size - n_train - n_val
        
        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:n_train + n_val]
        test_indices = all_indices[n_train + n_val:n_train + n_val + n_test]
        
        split_mode_str = "Chronological"
        
    else:
        raise ValueError(f"Unknown DATA_SPLIT_MODE: {split_mode}. Must be 'blocked_timeseries' or 'chronological'.")
    
    # Verify NO DATA LOSS
    total_split_size = len(train_indices) + len(val_indices) + len(test_indices)
    assert total_split_size == dataset_size, \
        f"Data loss detected: {len(train_indices)}+{len(val_indices)}+{len(test_indices)} = {total_split_size} != {dataset_size}"
    
    # Verify split ratios
    train_ratio = len(train_indices) / dataset_size
    val_ratio = len(val_indices) / dataset_size
    test_ratio = len(test_indices) / dataset_size
    expected_train = config.TRAIN_SPLIT
    expected_val = config.VAL_SPLIT
    expected_test = 1.0 - config.TRAIN_SPLIT - config.VAL_SPLIT
    
    print(f"[Data Split] Mode: {split_mode_str} | Train: {len(train_indices)} ({train_ratio:.1%}), Val: {len(val_indices)} ({val_ratio:.1%}), Test: {len(test_indices)} ({test_ratio:.1%}) | Expected: {expected_train:.1%}/{expected_val:.1%}/{expected_test:.1%} | Zero loss: {total_split_size}=={dataset_size} [OK]")
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Sequential models: NO shuffling (order matters)
    # Non-sequential models: CAN shuffle (order doesn't matter)
    shuffle_train = is_static  # True for GCN/adaptiveGCN, False for LSTM/GRU
    
    collate_fn_to_use = _collate_static if is_static else _collate_sequential_padded
    
    # OPTIMIZED DataLoader settings for professional GPU performance
    # pin_memory=True is CRITICAL for fast CPU->GPU data transfer (only when CUDA available)
    # Windows multiprocessing: Use num_workers=0 to avoid worker crashes
    import sys
    is_windows = sys.platform == 'win32'
    use_cuda = torch.cuda.is_available()
    
    # Get num_workers from config
    num_workers = config.NUM_WORKERS
    
    # On Windows, multiprocessing with DataLoader can cause worker crashes
    # Always use num_workers=0 on Windows for reliability (multiprocessing issues)
    if is_windows:
        if num_workers > 0:
            # Only print once per run (use module-level flag)
            if not hasattr(create_data_loaders, '_windows_warning_printed'):
                print(f"[DataLoader] Windows detected: Using num_workers=0 (was {num_workers}) to avoid multiprocessing crashes")
                create_data_loaders._windows_warning_printed = True
        num_workers = 0
    
    train_dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': shuffle_train,  # Dynamic based on model type
        'num_workers': num_workers,
        'collate_fn': collate_fn_to_use,
        'pin_memory': use_cuda,  # Only enable when CUDA is available
        'persistent_workers': num_workers > 0,  # Only if using workers
        'prefetch_factor': 2 if num_workers > 0 else None,  # Only if using workers
    }
    
    val_test_dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': False,  # Never shuffle val/test
        'num_workers': num_workers,
        'collate_fn': collate_fn_to_use,
        'pin_memory': use_cuda,  # Only enable when CUDA is available
        'persistent_workers': num_workers > 0,  # Only if using workers
        'prefetch_factor': 2 if num_workers > 0 else None,  # Only if using workers
    }
    
    train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **val_test_dataloader_kwargs)
    test_loader = DataLoader(test_dataset, **val_test_dataloader_kwargs)
    return train_loader, val_loader, test_loader