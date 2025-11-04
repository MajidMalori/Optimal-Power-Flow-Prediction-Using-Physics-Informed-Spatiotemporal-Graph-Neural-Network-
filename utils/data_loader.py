# In utils/data_loader.py

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
    
    CRITICAL: All targets are in consistent units for proper normalization:
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
        # Normalize features (measurements)
        self.feature_mean = np.mean(features, axis=(0, 1))  # [10]
        self.feature_std = np.std(features, axis=(0, 1))    # [10]
        self.feature_std[self.feature_std == 0] = 1.0
        
        # Normalize targets (voltages)
        self.target_mean = np.mean(targets, axis=(0, 1))    # [2]
        self.target_std = np.std(targets, axis=(0, 1))      # [2]
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
                data_cpu = data.cpu()
            else:
                data_cpu = data
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
        
        result = (data_np - mean_np) / std_np
        
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

        # PURE STATE ESTIMATION: Targets only have [vm, va] (2 features)
        # Extract generation components from FEATURES (measurements), not targets
        # Features format: [p_load, q_load, p_ext, q_ext, p_conv, q_conv, p_ren, q_ren, vm_meas, va_meas]
        if features_tensor.dim() == 3:
            # Sequential model: use last timestep
            features_last = features_tensor[-1] if features_tensor.shape[0] > 1 else features_tensor[0]
        else:
            features_last = features_tensor
        
        ext_grid_gen = features_last[:, 2:4]  # p_ext, q_ext (indices 2-3)
        conventional_gen = features_last[:, 4:6]  # p_conv, q_conv (indices 4-5)
        renewable_gen = features_last[:, 6:8]  # p_ren, q_ren (indices 6-7)
        
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
            'renewable_gen': renewable_gen
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
        print(f"\n[CRITICAL ERROR] Failed during adjacency matrix loading and conversion: {e}")
        raise

    all_features, all_ybus, all_targets = [], [], []
    all_bus_types = []  # OPF: Store bus types for each timestep
    all_energy_coeffs, all_carbon_coeffs = [], []
    all_renewable_fractions = []  # Track renewable fractions for each data file
    
    # Note: Generation components are now included in the features/targets matrices
    
    for f_path in feature_files:
        # --- START CORRECTION: Update filenames to match new saved data ---
        ybus_path = f_path.replace('features', 'ybus_matrices')
        # --- END CORRECTION ---
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
                print(f"[Warning] bus_types file not found: {bus_types_path}. Assuming all PQ buses.")
                all_bus_types.append(np.zeros((num_timesteps, num_buses), dtype=np.int32))
            
            all_energy_coeffs.append(np.loadtxt(energy_path))
            all_carbon_coeffs.append(np.loadtxt(carbon_path))
            
            # Note: Generation components are now included in the features/targets matrices
            
            # Create renewable fraction array for this data file
            renewable_fractions_for_file = np.full(features_data.shape[0], renewable_fraction)
            all_renewable_fractions.append(renewable_fractions_for_file)
        except FileNotFoundError as e:
            print(f"\n[CRITICAL ERROR] A required data file is missing: {e.filename}")
            print("Please ensure you have run 'gen_meas_best.py' to generate all necessary data files.")
            raise e

    # --- START CORRECTION: Concatenate all data arrays along the time axis ---
    concatenated_features = np.concatenate(all_features, axis=0)
    concatenated_targets = np.concatenate(all_targets, axis=0)
    concatenated_bus_types = np.concatenate(all_bus_types, axis=0) if all_bus_types else None
    concatenated_energy_coeffs = np.concatenate(all_energy_coeffs, axis=0)
    concatenated_carbon_coeffs = np.concatenate(all_carbon_coeffs, axis=0)
    concatenated_renewable_fractions = np.concatenate(all_renewable_fractions, axis=0)
    
    # Note: Generation components are now included in the features/targets matrices
    
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
    # --- END CORRECTION ---

    # Create normalizer with BOTH features and targets (for pure state estimation)
    normalizer = PowerSystemNormalizer(concatenated_features, concatenated_targets)
    features_norm = normalizer.normalize(concatenated_features)
    targets_norm = normalizer.normalize(concatenated_targets)  # CRITICAL: Normalize targets too!
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
    # This collate function will now work correctly as targets are already single slices.
    return default_collate(batch)

def _collate_sequential_padded(batch):
    # This collate function no longer needs to pad the targets.
    static_adj_matrix = batch[0]['adjacency']
    other_items_batch = [{k: v for k, v in item.items() if k not in ['features', 'targets', 'adjacency']} for item in batch]
    collated_batch = default_collate(other_items_batch)
    collated_batch['adjacency'] = static_adj_matrix
    features_list = [item['features'] for item in batch]
    collated_batch['features'] = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    # Targets are now a batch of [N, F] tensors, so we can stack them into [B, N, F]
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
    
    # TIME-SERIES mode: Use STRATIFIED split by renewable fraction for ALL models
    # This ensures ALL renewable fractions appear in train/val/test sets
    
    # Group indices by renewable fraction
    unique_fractions = np.unique(renewable_fractions)
    train_indices, val_indices, test_indices = [], [], []
    
    for frac in unique_fractions:
        # Get all sample indices for this renewable fraction
        frac_mask = renewable_fractions == frac
        frac_indices = np.where(frac_mask)[0]
        
        # Adjust for sequential models (some indices might be out of bounds)
        # Only keep indices that are valid for the dataset
        valid_frac_indices = [idx for idx in frac_indices if idx < dataset_size]
        
        if len(valid_frac_indices) == 0:
            continue
        
        # Split THIS fraction's data into train/val/test
        n_frac = len(valid_frac_indices)
        n_train = int(config.TRAIN_SPLIT * n_frac)
        n_val = int(config.VAL_SPLIT * n_frac)
        
        # Contiguous splits WITHIN each fraction (preserves temporal order)
        frac_train = valid_frac_indices[:n_train]
        frac_val = valid_frac_indices[n_train:n_train + n_val]
        frac_test = valid_frac_indices[n_train + n_val:]
        
        train_indices.extend(frac_train)
        val_indices.extend(frac_val)
        test_indices.extend(frac_test)
    
    # Sort indices to maintain some temporal structure across fractions
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Sequential models: NO shuffling (order matters)
    # Non-sequential models: CAN shuffle (order doesn't matter)
    shuffle_train = is_static  # True for GCN/adaptiveGCN, False for LSTM/GRU
    
    collate_fn_to_use = _collate_static if is_static else _collate_sequential_padded
    
    # OPTIMIZED DataLoader settings for memory efficiency
    train_dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': shuffle_train,  # Dynamic based on model type
        'num_workers': min(config.NUM_WORKERS, 4),  # Cap at 4 workers
        'collate_fn': collate_fn_to_use,
        'pin_memory': torch.cuda.is_available(),  # Pin memory for GPU
        'persistent_workers': True,  # Keep workers alive
        'prefetch_factor': 2,  # Prefetch 2 batches per worker
    }
    
    val_test_dataloader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': False,  # Never shuffle val/test
        'num_workers': min(config.NUM_WORKERS, 4),  # Cap at 4 workers
        'collate_fn': collate_fn_to_use,
        'pin_memory': torch.cuda.is_available(),  # Pin memory for GPU
        'persistent_workers': True,  # Keep workers alive
        'prefetch_factor': 2,  # Prefetch 2 batches per worker
    }
    
    train_loader = DataLoader(train_dataset, **train_dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **val_test_dataloader_kwargs)
    test_loader = DataLoader(test_dataset, **val_test_dataloader_kwargs)
    return train_loader, val_loader, test_loader