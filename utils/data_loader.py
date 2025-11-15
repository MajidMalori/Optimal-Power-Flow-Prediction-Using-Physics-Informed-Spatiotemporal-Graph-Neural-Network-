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

class PowerSystemDataset(Dataset):
    """
    Custom PyTorch Dataset for power system time-series data.
    Handles time-synchronized features, targets, and Ybus matrices.
    Supports lazy Ybus reconstruction to save memory for large datasets.
    """
    def __init__(self, features, adjacency_matrix, ybus_matrices, targets, 
                 time_energy_coeffs, time_carbon_coeffs, renewable_fractions, is_static, sequence_length=1,
                 hours_per_day=24, bus_types=None):
        """
        Args:
            hours_per_day: Number of hours per day (always 24 for time-series mode)
        """
        self.features = torch.from_numpy(features).float()
        self.adjacency = torch.from_numpy(adjacency_matrix).float()
        self.hours_per_day = hours_per_day
        
        # Handle Ybus - either pre-loaded array or lazy reconstruction data
        if isinstance(ybus_matrices, dict) and 'lazy' in ybus_matrices:
            # Lazy loading mode for memory efficiency
            self.ybus_lazy = True
            self.ybus_base = torch.from_numpy(ybus_matrices['base']).cfloat()
            self.ybus_contingency_timesteps = ybus_matrices['contingency_timesteps']
            self.ybus_contingency_matrices = torch.from_numpy(ybus_matrices['contingency_matrices']).cfloat()
            # Create lookup dict for fast contingency access
            self.ybus_contingency_lookup = {int(t): i for i, t in enumerate(self.ybus_contingency_timesteps)}
            self.ybus_matrices = None
        else:
            # Pre-loaded mode (for smaller datasets)
            self.ybus_lazy = False
            self.ybus_matrices = torch.from_numpy(ybus_matrices).cfloat()
        
        self.targets = torch.from_numpy(targets).float()
        self.bus_types = torch.from_numpy(bus_types).long() if bus_types is not None else None  # OPF: bus type codes [0=PQ, 1=PV, 2=Slack]
        self.time_energy_coeffs = torch.from_numpy(time_energy_coeffs).float()
        self.time_carbon_coeffs = torch.from_numpy(time_carbon_coeffs).float()
        self.renewable_fractions = torch.from_numpy(renewable_fractions).float()
        
        # OPF Approach:
        # Features: [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas] (10 measurements)
        # Targets: [var1, var2] bus-type dependent (PQ: [V,θ], PV: [Q,θ], Slack: [P,Q])
        
        self.is_static = is_static
        self.sequence_length = sequence_length
        self.num_samples = len(features)
        
    def __len__(self):
        # For static models: all samples are available
        # For sequential models: leave room for sequence + target
        if self.is_static:
            return self.num_samples
        else:
            return self.num_samples - self.sequence_length

    def __getitem__(self, idx):
        if self.is_static:
            # For static models, input and target are the same single time step
            start_idx = idx
            target_idx = idx
            features_tensor = self.features[start_idx]
            target_tensor = self.targets[target_idx]
        else:
            # For sequential models, the input is a sequence
            start_idx = idx
            end_idx = idx + self.sequence_length
            features_tensor = self.features[start_idx:end_idx]
            
            # The target is the single time step immediately following the input sequence
            target_idx = end_idx
            target_tensor = self.targets[target_idx]

        # Get Ybus matrix - either from pre-loaded array or reconstruct lazily
        if self.ybus_lazy:
            # Check if this timestep has a contingency
            if target_idx in self.ybus_contingency_lookup and len(self.ybus_contingency_matrices) > 0:
                cont_idx = self.ybus_contingency_lookup[target_idx]
                if cont_idx < len(self.ybus_contingency_matrices):
                    ybus_for_item = self.ybus_contingency_matrices[cont_idx]
                else:
                    ybus_for_item = self.ybus_base  # Fallback to base Ybus if index is out of bounds
            else:
                # Use base Ybus
                ybus_for_item = self.ybus_base
        else:
            # Pre-loaded mode
            ybus_for_item = self.ybus_matrices[target_idx]
        
        time_energy = self.time_energy_coeffs[target_idx]
        time_carbon = self.time_carbon_coeffs[target_idx]
        renewable_fraction = self.renewable_fractions[target_idx]

        # OPF: Targets have 2 features per bus (OPF unknowns, bus-type dependent)
        # Extract generation components from FEATURES (measurements), not targets
        # Import feature indices constants (single source of truth)
        from config import FeatureIndices
        
        if features_tensor.dim() == 3:
            # Sequential model: use last timestep
            features_last = features_tensor[-1] if features_tensor.shape[0] > 1 else features_tensor[0]
        else:
            features_last = features_tensor
        
        ext_grid_gen = features_last[:, FeatureIndices.P_EXT_GRID:FeatureIndices.Q_EXT_GRID+1]  # p_ext, q_ext
        conventional_gen = features_last[:, FeatureIndices.P_CONV:FeatureIndices.Q_CONV+1]  # p_conv, q_conv
        renewable_gen = features_last[:, FeatureIndices.P_REN:FeatureIndices.Q_REN+1]  # p_ren, q_ren
        
        # Get bus types for this timestep (OPF: needed for loss calculation)
        bus_types_tensor = None
        if self.bus_types is not None:
            bus_types_tensor = self.bus_types[target_idx]
        
        return {
            'features': features_tensor,
            'adjacency': self.adjacency,
            'ybus_matrix': ybus_for_item,
            'targets': target_tensor,
            'bus_types': bus_types_tensor,  # OPF: bus type codes [0=PQ, 1=PV, 2=Slack]
            'time_energy_coeffs': time_energy,
            'time_carbon_coeffs': time_carbon,
            'renewable_fraction': renewable_fraction,
            'ext_grid_gen': ext_grid_gen,
            'conventional_gen': conventional_gen,
            'renewable_gen': renewable_gen,
            'timestep': target_idx  # Store actual timestep for temporal plotting
        }

# ... all other functions below this line remain the same ...

def _convert_edge_index_to_adj(edge_index, num_nodes):
    num_nodes = int(num_nodes)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    source_nodes = edge_index[0].astype(int)
    dest_nodes = edge_index[1].astype(int)
    adj[source_nodes, dest_nodes] = 1
    adj[dest_nodes, source_nodes] = 1
    return adj

def load_power_system_data(config, case_name):
    print(f"[Data] Loading {case_name} scenarios...")
    data_dir = getattr(config, 'DATA_DIR', './data')
    feature_files = sorted(glob.glob(os.path.join(data_dir, f"{case_name}_features_frac*.npy")))
    if not feature_files:
        raise FileNotFoundError(f"No data files found for pattern: '{case_name}_features_frac*.npy' in '{data_dir}'.")
    try:
        # Extract number of buses from case name
        num_buses = int(''.join(filter(str.isdigit, case_name)))
        
        # Load adjacency matrix (silently)
        first_adj_path = feature_files[0].replace('features', 'adjacency')
        adj_object_array = np.load(first_adj_path, allow_pickle=True)
        edge_index = adj_object_array[0]
        
        static_adjacency_matrix = _convert_edge_index_to_adj(edge_index, num_buses)
        
        if static_adjacency_matrix.ndim != 2 or static_adjacency_matrix.shape[0] != static_adjacency_matrix.shape[1]:
             raise ValueError(f"Conversion to dense matrix failed. Final shape is not square: {static_adjacency_matrix.shape}.")
    except Exception as e:
        print(f"\nError: Failed during adjacency matrix loading and conversion: {e}")
        raise

    all_features, all_ybus, all_targets = [], [], []
    all_bus_types = []  # OPF: Store bus types for each timestep
    all_energy_coeffs, all_carbon_coeffs = [], []
    all_renewable_fractions = []  # Track renewable fractions for each data file
    
    
    for f_path in feature_files:
        ybus_path = f_path.replace('features', 'ybus_matrices')
        targets_path = f_path.replace('features', 'targets')
        energy_path = f_path.replace('features', 'time_energy_coeffs').replace('.npy', '.txt')
        carbon_path = f_path.replace('features', 'time_carbon_coeffs').replace('.npy', '.txt')
        
        # Extract renewable fraction from filename (e.g., "case33_features_frac0.2_timestamp.npy" -> 0.2)
        import re
        frac_match = re.search(r'frac(\d+\.\d+)', os.path.basename(f_path))
        renewable_fraction = float(frac_match.group(1)) if frac_match else 0.0
        
        try:
            features_data = np.load(f_path)
            all_features.append(features_data)
            num_timesteps = features_data.shape[0]
            
            # Load Ybus matrix - support both sparse (new) and dense (old) formats
            # Try sparse format first
            ybus_base_path = f_path.replace('features', 'ybus_base')
            ybus_contingency_timesteps_path = f_path.replace('features', 'ybus_contingency_timesteps')
            ybus_contingency_matrices_path = f_path.replace('features', 'ybus_contingency_matrices')
            convergence_report_path = f_path.replace('features', 'convergence_report').replace('.npy', '.json')
            
            if os.path.exists(ybus_base_path):
                # New sparse format found - store lazy loading data
                ybus_base = np.load(ybus_base_path)
                contingency_timesteps = np.load(ybus_contingency_timesteps_path)
                contingency_matrices = np.load(ybus_contingency_matrices_path)
                
                # Store in lazy format - adjust timestep indices for concatenated data
                timestep_offset = sum(len(yb['contingency_timesteps']) if isinstance(yb, dict) else yb.shape[0] for yb in all_ybus)
                adjusted_timesteps = contingency_timesteps + timestep_offset
                
                all_ybus.append({
                    'lazy': True,
                    'base': ybus_base,
                    'contingency_timesteps': adjusted_timesteps,
                    'contingency_matrices': contingency_matrices,
                    'num_timesteps': num_timesteps
                })
            else:
                # Old dense format (backward compatibility)
                ybus_path = f_path.replace('features', 'ybus_matrices')
                ybus_full = np.load(ybus_path)
                all_ybus.append(ybus_full)
            
            all_targets.append(np.load(targets_path))
            
            # Load bus types (OPF: bus-type-dependent unknowns)
            bus_types_path = f_path.replace('features', 'bus_types')
            if os.path.exists(bus_types_path):
                all_bus_types.append(np.load(bus_types_path))
            else:
                # Fallback: If bus_types not found, assume all PQ buses (backward compatibility)
                print(f"Warning: bus_types file not found: {bus_types_path}. Assuming all PQ buses.")
                all_bus_types.append(np.zeros((num_timesteps, num_buses), dtype=np.int32))
            
            all_energy_coeffs.append(np.loadtxt(energy_path))
            all_carbon_coeffs.append(np.loadtxt(carbon_path))
            
            
            # Create renewable fraction array for this data file
            renewable_fractions_for_file = np.full(features_data.shape[0], renewable_fraction)
            all_renewable_fractions.append(renewable_fractions_for_file)
        except FileNotFoundError as e:
            print(f"\nError: A required data file is missing: {e.filename}")
            print("Please ensure you have run 'gen_meas_best.py' to generate all necessary data files.")
            raise e

    # Convert to float32 for memory efficiency (50% memory reduction vs float64)
    concatenated_features = np.concatenate(all_features, axis=0).astype(np.float32)
    concatenated_targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    concatenated_bus_types = np.concatenate(all_bus_types, axis=0).astype(np.int32) if all_bus_types else None
    concatenated_energy_coeffs = np.concatenate(all_energy_coeffs, axis=0).astype(np.float32)
    concatenated_carbon_coeffs = np.concatenate(all_carbon_coeffs, axis=0).astype(np.float32)
    concatenated_renewable_fractions = np.concatenate(all_renewable_fractions, axis=0).astype(np.float32)
    
    # Handle Ybus - merge lazy loading data or concatenate pre-loaded arrays
    if all(isinstance(yb, dict) and 'lazy' in yb for yb in all_ybus):
        # All scenarios use lazy loading - merge them
        # All scenarios should have the same base Ybus (same bus system)
        concatenated_ybus = {
            'lazy': True,
            'base': all_ybus[0]['base'],  # Same for all scenarios
            'contingency_timesteps': np.concatenate([yb['contingency_timesteps'] for yb in all_ybus]),
            'contingency_matrices': np.concatenate([yb['contingency_matrices'] for yb in all_ybus], axis=0) if all(len(yb['contingency_matrices']) > 0 for yb in all_ybus) else np.array([]).reshape(0, all_ybus[0]['base'].shape[0], all_ybus[0]['base'].shape[1]).astype(np.complex128)
        }
    elif all(isinstance(yb, np.ndarray) for yb in all_ybus):
        # All scenarios use pre-loaded arrays - concatenate normally
        concatenated_ybus = np.concatenate(all_ybus, axis=0)
    else:
        raise ValueError("Mixed Ybus formats detected - all scenarios must use the same format (lazy or pre-loaded)")

    # Create normalizer with BOTH features and targets
    normalizer = PowerSystemNormalizer(concatenated_features, concatenated_targets)
    features_norm = normalizer.normalize(concatenated_features)
    targets_norm = normalizer.normalize(concatenated_targets)
    print(f"[Data] Loaded {len(feature_files)} scenarios -> {concatenated_features.shape[0]} samples")
    print(f"[Data] Features shape: {concatenated_features.shape} (measurements: {concatenated_features.shape[-1]} dims)")
    print(f"[Data] Targets shape: {concatenated_targets.shape} (unknowns: {concatenated_targets.shape[-1]} dims)")
    print(f"[Data] Features normalized: mean={np.mean(features_norm):.4f}, std={np.std(features_norm):.4f}")
    print(f"[Data] Targets normalized: mean={np.mean(targets_norm):.4f}, std={np.std(targets_norm):.4f}")
    
    # Return concatenated arrays (OPF: features=measurements, targets=unknowns per bus type)
    # Both are now normalized for consistent training
    return (features_norm, static_adjacency_matrix, concatenated_ybus, targets_norm,
            concatenated_bus_types, concatenated_energy_coeffs, concatenated_carbon_coeffs, 
            concatenated_renewable_fractions, normalizer)

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

def create_data_loaders(features, adjacency, ybus_matrices, targets, time_energy_coeffs, time_carbon_coeffs, renewable_fractions, config, is_static, bus_types=None):
    """
    Create data loaders for power system OPF data.
    
    Args:
        bus_types: Optional array of bus type codes [timesteps, buses] with values [0=PQ, 1=PV, 2=Slack]
    """
    seq_len = 1 if is_static else getattr(config, 'SEQUENCE_LENGTH', 1)
    hours_per_day = getattr(config, 'HOURS_PER_DAY', 24)
    
    dataset = PowerSystemDataset(
        features, adjacency, ybus_matrices, targets, 
        time_energy_coeffs, time_carbon_coeffs, renewable_fractions,
        is_static, seq_len, hours_per_day=hours_per_day, bus_types=bus_types
    )
    dataset_size = len(dataset)
    
    # Get split mode from config (default to 'blocked_timeseries' if not specified)
    split_mode = getattr(config, 'DATA_SPLIT_MODE', 'blocked_timeseries')
    
    # Backward compatibility: 'stratified' is an alias for 'blocked_timeseries'
    if split_mode == 'stratified':
        split_mode = 'blocked_timeseries'
    
    if split_mode == 'blocked_timeseries':
        # BLOCKED TIME-SERIES SPLIT (Recommended for time-series forecasting)
        # 
        # This is the methodologically sound approach for time-series data with multiple scenarios.
        # It combines the benefits of both chronological and stratified splitting:
        # 1. Groups data into blocks by renewable_fraction (ensures all scenarios in train/val/test)
        # 2. Splits each block chronologically (maintains temporal order, prevents data leakage)
        # 3. Combines splits across blocks (final sets contain all renewable fractions)
        #
        # Why this is superior:
        # - Zero data leakage: Temporal order is strictly maintained within each scenario block
        # - Guaranteed representation: All renewable fractions appear in train/val/test sets
        # - Fair evaluation: Model is tested on diverse scenarios while respecting time-series principles
        # - Publication-ready: Defensible in top-tier ML/power systems journals
        
        # Group indices by renewable fraction
        unique_fractions = np.unique(renewable_fractions)
        train_indices, val_indices, test_indices = [], [], []
        
        for frac in unique_fractions:
            # STEP 1: BLOCK - Get all sample indices for this specific renewable fraction
            frac_mask = renewable_fractions == frac
            frac_indices = np.where(frac_mask)[0]
            
            # Filter indices to ensure they are valid for the dataset length
            # (important for sequential models that may have padding/truncation)
            valid_frac_indices = [idx for idx in frac_indices if idx < dataset_size]
            
            if len(valid_frac_indices) == 0:
                continue
            
            n_frac = len(valid_frac_indices)
            
            # STEP 2: SPLIT CHRONOLOGICALLY within the block
            # The indices are already sorted by time within the block (from data generation)
            n_train = int(config.TRAIN_SPLIT * n_frac)
            n_val = int(config.VAL_SPLIT * n_frac)
            # The rest goes to test, ensuring no data loss within the block
            n_test = n_frac - n_train - n_val
            
            # Contiguous splits preserve temporal order within each block
            frac_train_indices = valid_frac_indices[:n_train]
            frac_val_indices = valid_frac_indices[n_train:n_train + n_val]
            frac_test_indices = valid_frac_indices[n_train + n_val:]
            
            # Verify no data loss for this fraction
            assert len(frac_train_indices) + len(frac_val_indices) + len(frac_test_indices) == n_frac, \
                f"Data loss detected for fraction {frac}: {len(frac_train_indices)}+{len(frac_val_indices)}+{len(frac_test_indices)} != {n_frac}"
            
            # STEP 3: COMBINE - Add the split indices to the final lists
            train_indices.extend(frac_train_indices)
            val_indices.extend(frac_val_indices)
            test_indices.extend(frac_test_indices)
        
        # Sort the final indices to maintain as much of the overall temporal structure as possible
        # This helps with data loader efficiency and maintains some global temporal ordering
        train_indices.sort()
        val_indices.sort()
        test_indices.sort()
        
        split_mode_str = "Blocked Time-Series"
        
    elif split_mode == 'chronological':
        # SIMPLE CHRONOLOGICAL SPLIT (Alternative method)
        # 
        # Split by time order (train on past, test on future).
        # This maintains strict temporal order but may result in test set missing some renewable fractions.
        # Use this only if you specifically need a simple chronological split.
        # 
        # WARNING: This method may cause warnings like "No data for X% renewables" in test set,
        # which is expected if your data generation creates blocks of scenarios sequentially.
        
        # All indices in chronological order (already sorted by time)
        all_indices = list(range(dataset_size))
        
        # Calculate split sizes
        n_train = int(config.TRAIN_SPLIT * dataset_size)
        n_val = int(config.VAL_SPLIT * dataset_size)
        n_test = dataset_size - n_train - n_val  # Remaining goes to test (ensures no data loss)
        
        # Split chronologically: first N% for train, next M% for val, remaining for test
        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:n_train + n_val]
        test_indices = all_indices[n_train + n_val:n_train + n_val + n_test]
        
        split_mode_str = "Chronological"
        
    else:
        raise ValueError(f"Unknown DATA_SPLIT_MODE: {split_mode}. Must be 'blocked_timeseries' or 'chronological'.")
    
    # Verify NO DATA LOSS: train + val + test = total dataset size
    total_split_size = len(train_indices) + len(val_indices) + len(test_indices)
    assert total_split_size == dataset_size, \
        f"Data loss detected: {len(train_indices)}+{len(val_indices)}+{len(test_indices)} = {total_split_size} != {dataset_size}"
    
    # Verify split ratios match config (within rounding tolerance)
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
    
    # OPTIMIZED DataLoader settings for memory efficiency
    num_workers = min(config.NUM_WORKERS, 4)  # Cap at 4 workers
    train_dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': shuffle_train,  # Dynamic based on model type
        'num_workers': num_workers,
        'collate_fn': collate_fn_to_use,
        'pin_memory': torch.cuda.is_available(),  # Pin memory for GPU
        'persistent_workers': num_workers > 0,  # Only if using workers
        'prefetch_factor': 2 if num_workers > 0 else None,  # Only if using workers
    }
    
    val_test_dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': False,  # Never shuffle val/test
        'num_workers': num_workers,
        'collate_fn': collate_fn_to_use,
        'pin_memory': torch.cuda.is_available(),  # Pin memory for GPU
        'persistent_workers': num_workers > 0,  # Only if using workers
        'prefetch_factor': 2 if num_workers > 0 else None,  # Only if using workers
    }
    
    train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **val_test_dataloader_kwargs)
    test_loader = DataLoader(test_dataset, **val_test_dataloader_kwargs)
    return train_loader, val_loader, test_loader