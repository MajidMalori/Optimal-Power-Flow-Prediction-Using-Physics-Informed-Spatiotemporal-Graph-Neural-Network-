import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

class PowerSystemNormalizer:
    """
    A class to handle normalization and de-normalization of power system features using Global Per-Unit scaling.
    
    New Logic (Global Per-Unit):
    - Active/Reactive Power (Cols 0-7): val / base_mva
    - Voltage Mag (Col 8): (val - 1.0) * 10.0
    - Voltage Angle (Col 9): val * 10.0
    
    Targets (10-dim) are normalized exactly the same way as features.
    """
    def __init__(self, features, targets, base_mva=100.0):
        """
        Args:
            features: Input measurements [samples, buses, 10]
            targets: Clean state [samples, buses, 10]
            base_mva: System base power (MVA)
        """
        self.base_mva = float(base_mva)
        
        # Define normalization parameters
        # Power scaling: 1.0 / base_mva
        self.power_scale = 1.0 / self.base_mva
        
        # Voltage scaling: Center at 1.0, scale by 10.0
        self.vm_center = 1.0
        self.vm_scale = 10.0
        
        # Angle scaling: Scale by 10.0
        self.va_scale = 10.0
        
        # Store dummy stats for compatibility if needed (though we don't use z-score anymore)
        self.feature_mean = np.zeros(10, dtype=np.float32)
        self.feature_std = np.ones(10, dtype=np.float32)
        self.target_mean = np.zeros(10, dtype=np.float32)
        self.target_std = np.ones(10, dtype=np.float32)

    def normalize(self, data):
        """
        Normalize data using Global Per-Unit scaling.
        Works for both Features and Targets (since both are 10-dim state vectors).
        
        Args:
            data: Input data [samples, buses, 10] or tensor equivalent
            
        Returns:
            Normalized data in same format as input
        """
        # Handle both numpy arrays and PyTorch tensors
        if torch.is_tensor(data):
            if data.is_cuda:
                data_cpu = data.detach().cpu()
            else:
                data_cpu = data.detach().cpu()
            data_np = data_cpu.numpy()
            was_tensor = True
            original_device = data.device if data.is_cuda else None
        else:
            data_np = data
            was_tensor = False
            original_device = None
        
        # Ensure float32
        data_np = data_np.astype(np.float32)
        
        result = np.zeros_like(data_np)
        
        # Check dimensions
        if data_np.shape[-1] != 10:
            # Fallback for partial data or 2-dim targets (legacy support or error?)
            # If legacy 2-dim targets [V, theta] or [P, Q] etc., this scaler might not apply correctly
            # without knowing WHAT the columns are.
            # But we assume 10-dim now.
            # If 2-dim (OPF unknowns), we can't easily apply this global scaling without bus types.
            # Assuming data is 10-dim.
            pass
            
        # Apply scaling
        # Cols 0-7: Power (P/Q) -> val / base_mva
        result[..., 0:8] = data_np[..., 0:8] * self.power_scale
        
        # Col 8: Voltage Mag -> (val - 1.0) * 10.0
        result[..., 8] = (data_np[..., 8] - self.vm_center) * self.vm_scale
        
        # Col 9: Voltage Angle -> val * 10.0
        result[..., 9] = data_np[..., 9] * self.va_scale
        
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
        Exact inverse of normalize.
        
        Args:
            data: Tensor of shape [..., 10]
            
        Returns:
            Denormalized tensor
        """
        # Handle both torch tensors and numpy arrays
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = torch.from_numpy(data).float()
        
        device = data.device
        
        # Create result tensor
        result = torch.zeros_like(data)
        
        # Cols 0-7: Power (P/Q) -> val * base_mva
        result[..., 0:8] = data[..., 0:8] / self.power_scale
        
        # Col 8: Voltage Mag -> (val / 10.0) + 1.0
        result[..., 8] = (data[..., 8] / self.vm_scale) + self.vm_center
        
        # Col 9: Voltage Angle -> val / 10.0
        result[..., 9] = data[..., 9] / self.va_scale
        
        # Return numpy if input was numpy
        if is_numpy:
            return result.cpu().numpy()
        return result

    def denormalize_targets(self, data: torch.Tensor) -> torch.Tensor:
        """Wrapper for denormalizing targets (same as features now)."""
        return self.denormalize(data)

class PowerSystemLazyDataset(Dataset):
    """
    Professional lazy-loading PyTorch Dataset for power system time-series data.
    
    This dataset only stores file paths and metadata, loading data on-demand in __getitem__.
    This is the scalable, memory-efficient approach for large datasets.
    """
    def __init__(self, file_metadata, adjacency_matrix, normalizer, ybus_metadata, 
                 is_static, sequence_length=1, hours_per_day=24, topology_cache=None, topology_ids=None):
        """
        Args:
            file_metadata: List of dicts, each containing file paths and metadata for one sample
            adjacency_matrix: Base adjacency matrix [num_buses, num_buses]
            normalizer: PowerSystemNormalizer instance
            ybus_metadata: Dict with 'base_path' etc.
            is_static: Whether this is a static model (single timestep) or sequential
            sequence_length: Length of input sequence
            hours_per_day: Number of hours per day
            topology_cache: Dict mapping topology_id -> pre-normalized adjacency tensor
            topology_ids: Array mapping sample index -> topology_id
        """
        self.file_metadata = file_metadata
        self.adjacency_matrix = adjacency_matrix
        self.normalizer = normalizer
        self.ybus_metadata = ybus_metadata
        self.is_static = is_static
        self.sequence_length = sequence_length
        self.hours_per_day = hours_per_day
        
        # Load base Ybus matrix (REQUIRED)
        if ybus_metadata and 'base_path' in ybus_metadata and os.path.exists(ybus_metadata['base_path']):
            self.ybus_base = torch.from_numpy(np.load(ybus_metadata['base_path'], mmap_mode='r').copy()).cfloat()
        else:
            raise RuntimeError(f"Ybus base matrix not found. Required for physics-informed training.")
            
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
            # Load features
            features = np.load(target_meta['features_path'], mmap_mode='r')[target_idx_in_file].copy()
            features_tensor = torch.from_numpy(features).float()
            
            # Load targets (Full Clean State 10-dim)
            targets = np.load(target_meta['targets_path'], mmap_mode='r')[target_idx_in_file].copy()
            target_tensor = torch.from_numpy(targets).float()
            
            # Normalize on-the-fly
            features_tensor = self.normalizer.normalize(features_tensor.unsqueeze(0)).squeeze(0)
            target_tensor = self.normalizer.normalize(target_tensor.unsqueeze(0)).squeeze(0)
            
        else:
            # Sequential model
            start_idx = idx
            end_idx = idx + self.sequence_length
            target_idx = end_idx
            
            # Load features sequence
            features_list = []
            for i in range(start_idx, end_idx):
                if i < len(self.file_metadata):
                    meta = self.file_metadata[i]
                    features = np.load(meta['features_path'], mmap_mode='r')[meta['index_in_file']].copy()
                    features_list.append(features)
            
            # Load target (next timestep)
            if target_idx < len(self.file_metadata):
                target_meta = self.file_metadata[target_idx]
                targets = np.load(target_meta['targets_path'], mmap_mode='r')[target_meta['index_in_file']].copy()
                target_tensor = torch.from_numpy(targets).float()
            else:
                raise IndexError(f"Target index {target_idx} out of bounds")
            
            features_array = np.stack(features_list, axis=0)
            features_tensor = torch.from_numpy(features_array).float()
            
            # Normalize
            features_tensor = self.normalizer.normalize(features_tensor)
            target_tensor = self.normalizer.normalize(target_tensor.unsqueeze(0)).squeeze(0)
        
        # Get Ybus matrix
        if self.ybus_base is not None:
            file_contingency_path = target_meta.get('ybus_contingency_matrices_path', None)
            cont_local_idx = target_meta.get('ybus_contingency_local_idx', None)
            
            if file_contingency_path and cont_local_idx is not None and os.path.exists(file_contingency_path):
                contingency_matrices = np.load(file_contingency_path, mmap_mode='r')
                if 0 <= cont_local_idx < contingency_matrices.shape[0]:
                    ybus_for_item = torch.from_numpy(contingency_matrices[cont_local_idx].copy()).cfloat()
                else:
                    raise IndexError(f"Contingency index {cont_local_idx} out of bounds")
            else:
                ybus_for_item = self.ybus_base
        else:
            raise RuntimeError("Ybus base matrix not available")
        
        # Load coefficients
        energy_coeffs = np.loadtxt(target_meta['energy_path'])
        carbon_coeffs = np.loadtxt(target_meta['carbon_path'])
        time_energy = energy_coeffs[target_meta['index_in_file']] if len(energy_coeffs) > target_meta['index_in_file'] else energy_coeffs[0]
        time_carbon = carbon_coeffs[target_meta['index_in_file']] if len(carbon_coeffs) > target_meta['index_in_file'] else carbon_coeffs[0]
        
        renewable_fraction = target_meta['renewable_fraction']
        
        # Load bus types
        bus_types_tensor = None
        if target_meta.get('bus_types_path') and os.path.exists(target_meta['bus_types_path']):
            bus_types = np.load(target_meta['bus_types_path'], mmap_mode='r')
            if bus_types.ndim == 2:
                bus_types_tensor = torch.from_numpy(bus_types[target_meta['index_in_file']].copy()).long()
            else:
                bus_types_tensor = torch.from_numpy(bus_types.copy()).long()
        
        # Extract generation components (from FEATURES, which are noisy measurements)
        from config import FeatureIndices
        if features_tensor.dim() == 3:
            features_last = features_tensor[-1]
        else:
            features_last = features_tensor
        
        ext_grid_gen = features_last[:, FeatureIndices.P_EXT_GRID:FeatureIndices.Q_EXT_GRID+1]
        conventional_gen = features_last[:, FeatureIndices.P_CONV:FeatureIndices.Q_CONV+1]
        renewable_gen = features_last[:, FeatureIndices.P_REN:FeatureIndices.Q_REN+1]
        
        # Topology cache
        if self.topology_cache is None or self.topology_ids is None:
            raise RuntimeError("Topology cache not initialized.")
        
        topology_id = self.topology_ids[idx]
        if topology_id not in self.topology_cache:
            raise KeyError(f"Topology ID {topology_id} not found in cache.")
        
        adjacency_for_item = self.topology_cache[topology_id]
        
        return {
            'features': features_tensor,
            'adjacency': adjacency_for_item,
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
    import torch
    adj_tensor = torch.from_numpy(adj).float()
    num_nodes = adj_tensor.shape[0]
    identity = torch.eye(num_nodes, dtype=adj_tensor.dtype)
    adj_hat = adj_tensor + identity
    degree = torch.sum(adj_hat, dim=1)
    epsilon = 1e-8
    degree = degree + epsilon
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt = torch.clamp(degree_inv_sqrt, min=0.0, max=1e10)
    degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
    normalized_adj = degree_matrix_inv_sqrt @ adj_hat @ degree_matrix_inv_sqrt
    return normalized_adj.numpy().astype(np.float32)

def _build_topology_cache_from_ids(file_metadata, base_adjacency, num_buses, case_name, data_dir, disable_normalization=False):
    """
    Builds a cache of adjacency matrices for each unique topology ID.
    
    Args:
        file_metadata: List of metadata dictionaries
        base_adjacency: Base adjacency matrix [num_buses, num_buses]
        num_buses: Number of buses
        case_name: Case name (e.g., 'case118')
        data_dir: Directory containing data files
        disable_normalization: If True, returns RAW adjacency (no pre-normalization).
                              Models must normalize internally.
    """
    topology_ids = np.array([meta.get('topology_id', 0) for meta in file_metadata], dtype=np.int32)
    unique_topology_ids = np.unique(topology_ids)
    
    topology_cache = {}
    
    # Base Adjacency
    if disable_normalization:
        # Store RAW adjacency
        topology_cache[0] = torch.from_numpy(base_adjacency).float()
    else:
        # Store PRE-NORMALIZED adjacency
        normalized_base = pre_normalize_adjacency(base_adjacency)
        topology_cache[0] = torch.from_numpy(normalized_base).float()
    
    if len(unique_topology_ids) > 1:
        try:
            import pandapower.networks as pn
        except ImportError:
            raise ImportError("pandapower is required to build contingency topologies.")
        
        if '33' in case_name:
            net = pn.case33bw()
        elif '57' in case_name:
            net = pn.case57()
        elif '118' in case_name:
            net = pn.case118()
        else:
            raise ValueError(f"Unknown case name: {case_name}")
        
        from utils.contingency_ybus import modify_adjacency_for_line_outage
        
        for topo_id in unique_topology_ids:
            if topo_id > 0:
                line_idx = topo_id - 1
                if line_idx not in net.line.index:
                    raise IndexError(f"Line index {line_idx} not found in network.")
                
                contingency_adj = modify_adjacency_for_line_outage(base_adjacency, net, line_idx)
                
                if disable_normalization:
                     # Store RAW adjacency (just contingency modifications)
                     topology_cache[topo_id] = torch.from_numpy(contingency_adj).float()
                else:
                     # Store PRE-NORMALIZED adjacency
                     normalized_cont = pre_normalize_adjacency(contingency_adj)
                     topology_cache[topo_id] = torch.from_numpy(normalized_cont).float()
    
    return topology_cache, topology_ids

def load_power_system_data(config, case_name):
    print(f"[Data] Creating lazy data manifest for {case_name}...", end=" ", flush=True)
    data_dir = getattr(config, 'DATA_DIR', './data')
    feature_files = sorted(glob.glob(os.path.join(data_dir, f"{case_name}_features_frac*.npy")))
    if not feature_files:
        raise FileNotFoundError(f"No data files found for pattern: '{case_name}_features_frac*.npy' in '{data_dir}'.")
    
    try:
        num_buses = int(''.join(filter(str.isdigit, case_name)))
        
        # Determine Base MVA based on case name
        if num_buses <= 33:
            base_mva = 10.0
        else:
            base_mva = 100.0
        
        first_features_path = feature_files[0]
        base_adj_path = first_features_path.replace('features', 'base_adjacency')
        
        if not os.path.exists(base_adj_path):
            raise FileNotFoundError(f"REQUIRED: base_adjacency file not found at {base_adj_path}")
        
        adj_object_array = np.load(base_adj_path, allow_pickle=True)
        edge_index = adj_object_array[0]
        raw_base_adjacency = _convert_edge_index_to_adj(edge_index, num_buses)
        base_adjacency_matrix = raw_base_adjacency
    except Exception as e:
        print(f"\nError: Failed during adjacency matrix loading: {e}")
        raise
    
    file_metadata = []
    global_timestep = 0
    ybus_base_path = None
    
    import re
    for f_path in feature_files:
        targets_path = f_path.replace('features', 'targets')
        bus_types_path = f_path.replace('features', 'bus_types')
        energy_path = f_path.replace('features', 'time_energy_coeffs').replace('.npy', '.txt')
        carbon_path = f_path.replace('features', 'time_carbon_coeffs').replace('.npy', '.txt')
        
        frac_match = re.search(r'frac(\d+\.\d+)', os.path.basename(f_path))
        renewable_fraction = float(frac_match.group(1)) if frac_match else 0.0
        
        features_mmap = np.load(f_path, mmap_mode='r', allow_pickle=False)
        num_timesteps = features_mmap.shape[0]
        
        if ybus_base_path is None:
            ybus_base_path = f_path.replace('features', 'ybus_base')
        
        file_contingency_matrices_path = f_path.replace('features', 'ybus_contingency_matrices')
        file_contingency_timesteps_path = f_path.replace('features', 'ybus_contingency_timesteps')
        
        file_topology_ids_path = f_path.replace('features', 'topology_ids')
        if not os.path.exists(file_topology_ids_path):
            raise FileNotFoundError(f"REQUIRED: topology_ids file not found at {file_topology_ids_path}")
        
        file_topology_ids = np.load(file_topology_ids_path)
        
        file_contingency_local_indices = {}
        if os.path.exists(file_contingency_timesteps_path):
            contingency_timesteps = np.load(file_contingency_timesteps_path)
            for local_cont_idx, local_ts in enumerate(contingency_timesteps):
                file_contingency_local_indices[int(local_ts)] = local_cont_idx
        
        for i in range(num_timesteps):
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
                'topology_id': int(topology_id),
                'ybus_contingency_matrices_path': file_contingency_matrices_path if os.path.exists(file_contingency_matrices_path) else None,
                'ybus_contingency_local_idx': file_contingency_local_indices.get(i, None)
            }
            file_metadata.append(entry)
            global_timestep += 1
    
    # Create normalizer with explicit Base MVA
    normalizer = PowerSystemNormalizer(None, None, base_mva=base_mva)
    
    ybus_metadata = {}
    if ybus_base_path and os.path.exists(ybus_base_path):
        ybus_metadata['base_path'] = ybus_base_path
    else:
        ybus_metadata = None
    
    print(f"done. Manifest: {len(file_metadata)} samples | Features: 10-dim | Targets: 10-dim (Clean) | Base MVA: {base_mva}", flush=True)
    
    print(f"[Data] Building topology cache...", end=" ", flush=True)
    # We now DISABLE pre-normalization in the data loader.
    # Models must handle normalization (adding self-loops, etc.) internally.
    # This prevents "Double-Dip Normalization" issues.
    
    # Pass raw adjacency to topology cache builder
    topology_cache, topology_ids_array = _build_topology_cache_from_ids(
        file_metadata, base_adjacency_matrix, num_buses, case_name, data_dir,
        disable_normalization=True # NEW FLAG: Raw adjacency returned
    )
    
    return (file_metadata, base_adjacency_matrix, ybus_metadata, normalizer, topology_cache, topology_ids_array)
def _collate_static(batch):
    static_adj_matrix = batch[0]['adjacency']
    batch_size = len(batch)
    
    if static_adj_matrix.dim() == 2:
        static_adj_batch = static_adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
    elif static_adj_matrix.dim() == 3:
        static_adj_batch = static_adj_matrix[0].unsqueeze(0).expand(batch_size, -1, -1)
    else:
        static_adj_batch = static_adj_matrix
    
    other_items_batch = [{k: v for k, v in item.items() if k != 'adjacency'} for item in batch]
    collated_batch = default_collate(other_items_batch)
    collated_batch['adjacency'] = static_adj_batch
    return collated_batch

def _collate_sequential_padded(batch):
    static_adj_matrix = batch[0]['adjacency']
    batch_size = len(batch)
    
    if static_adj_matrix.dim() == 2:
        static_adj_batch = static_adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
    elif static_adj_matrix.dim() == 3:
        static_adj_batch = static_adj_matrix[0].unsqueeze(0).expand(batch_size, -1, -1)
    else:
        static_adj_batch = static_adj_matrix
    
    other_items_batch = [{k: v for k, v in item.items() if k not in ['features', 'targets', 'adjacency']} for item in batch]
    collated_batch = default_collate(other_items_batch)
    collated_batch['adjacency'] = static_adj_batch
    
    features_list = [item['features'] for item in batch]
    collated_batch['features'] = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    collated_batch['targets'] = default_collate([item['targets'] for item in batch])
    
    return collated_batch

def create_data_loaders(file_metadata, adjacency, ybus_metadata, normalizer, config, is_static, 
                        topology_cache=None, topology_ids=None):
    seq_len = 1 if is_static else getattr(config, 'SEQUENCE_LENGTH', 1)
    hours_per_day = getattr(config, 'HOURS_PER_DAY', 24)
    
    dataset = PowerSystemLazyDataset(
        file_metadata, adjacency, normalizer, ybus_metadata,
        is_static, seq_len, hours_per_day=hours_per_day,
        topology_cache=topology_cache, topology_ids=topology_ids
    )
    dataset_size = len(dataset)
    
    renewable_fractions = np.array([meta['renewable_fraction'] for meta in file_metadata])
    split_mode = getattr(config, 'DATA_SPLIT_MODE', 'blocked_timeseries')
    if split_mode == 'stratified':
        split_mode = 'blocked_timeseries'
    
    if split_mode == 'blocked_timeseries':
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
        
    elif split_mode == 'chronological':
        all_indices = list(range(dataset_size))
        n_train = int(config.TRAIN_SPLIT * dataset_size)
        n_val = int(config.VAL_SPLIT * dataset_size)
        n_test = dataset_size - n_train - n_val
        
        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:n_train + n_val]
        test_indices = all_indices[n_train + n_val:n_train + n_val + n_test]
        
    else:
        raise ValueError(f"Unknown DATA_SPLIT_MODE: {split_mode}")
    
    total_split_size = len(train_indices) + len(val_indices) + len(test_indices)
    assert total_split_size == dataset_size, "Data loss detected in split"
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    shuffle_train = is_static
    collate_fn_to_use = _collate_static if is_static else _collate_sequential_padded
    
    import sys
    is_windows = sys.platform == 'win32'
    use_cuda = torch.cuda.is_available()
    num_workers = config.NUM_WORKERS
    
    if is_windows:
        num_workers = 0
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle_train, 
                              num_workers=num_workers, collate_fn=collate_fn_to_use, 
                              pin_memory=use_cuda, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn_to_use, 
                            pin_memory=use_cuda, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                             num_workers=num_workers, collate_fn=collate_fn_to_use, 
                             pin_memory=use_cuda, persistent_workers=(num_workers > 0))
    return train_loader, val_loader, test_loader

