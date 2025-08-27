# In utils/data_loader.py

import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

class PowerSystemNormalizer:
    """A class to handle normalization and de-normalization of power system features."""
    def __init__(self, features):
        self.mean = np.mean(features, axis=(0, 1))
        self.std = np.std(features, axis=(0, 1))
        self.std[self.std == 0] = 1.0

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data: torch.Tensor, num_buses: int) -> torch.Tensor:
        original_shape = data.shape
        if data.dim() == 3:
            num_output_features = data.shape[-1]
            data_reshaped = data
        elif data.dim() == 2:
            if num_buses <= 0: raise ValueError("num_buses must be positive for 2D tensor denormalization.")
            num_output_features = data.shape[-1] // num_buses
            if num_output_features == 0:
                total_features = data.shape[-1]
                expected_features_per_bus = 6
                if total_features % expected_features_per_bus == 0:
                    actual_num_buses = total_features // expected_features_per_bus
                    raise ValueError(f"Shape mismatch: Expected {num_buses} buses but model output suggests {actual_num_buses} buses. "
                                   f"Data shape: {data.shape}. This may indicate a model architecture issue.")
                else:
                    raise ValueError(f"Invalid tensor shape: {data.shape} for {num_buses} buses. Cannot determine features per bus.")
            data_reshaped = data.view(-1, num_buses, num_output_features)
        else:
            raise ValueError(f"denormalize expects a 2D or 3D tensor, but got {data.dim()}D.")

        mean_slice = self.mean[:num_output_features]
        std_slice = self.std[:num_output_features]
        mean_tensor = torch.from_numpy(mean_slice).float().to(data.device)
        std_tensor = torch.from_numpy(std_slice).float().to(data.device)

        denormalized_data = data_reshaped * std_tensor + mean_tensor
        return denormalized_data.view(original_shape)

class PowerSystemDataset(Dataset):
    """
    Custom PyTorch Dataset for power system time-series data.
    Handles time-synchronized features, targets, and Ybus matrices.
    """
    def __init__(self, features, adjacency_matrix, ybus_matrices, targets, 
                 time_energy_coeffs, time_carbon_coeffs, is_static, sequence_length=1):
        
        self.features = torch.from_numpy(features).float()
        self.adjacency = torch.from_numpy(adjacency_matrix).float()
        # --- START CORRECTION: Handle the concatenated Ybus array ---
        self.ybus_matrices = torch.from_numpy(ybus_matrices).cfloat()
        # --- END CORRECTION ---
        self.targets = torch.from_numpy(targets).float()
        self.time_energy_coeffs = torch.from_numpy(time_energy_coeffs).float()
        self.time_carbon_coeffs = torch.from_numpy(time_carbon_coeffs).float()
        
        self.is_static = is_static
        self.sequence_length = sequence_length
        self.num_samples = len(features)
        
    def __len__(self):
        # The last possible start_idx must leave room for one target time step after the sequence.
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

        # --- START CORRECTION: Select the Ybus matrix corresponding to the target time step ---
        ybus_for_item = self.ybus_matrices[target_idx]
        # --- END CORRECTION ---
        time_energy = self.time_energy_coeffs[target_idx]
        time_carbon = self.time_carbon_coeffs[target_idx]

        return {
            'features': features_tensor,
            'adjacency': self.adjacency,
            'ybus_matrix': ybus_for_item,
            'targets': target_tensor,
            'time_energy_coeffs': time_energy,
            'time_carbon_coeffs': time_carbon,
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
    print(f"[Data] Finding and loading all data scenarios for {case_name}...")
    data_dir = getattr(config, 'DATA_DIR', './data')
    feature_files = sorted(glob.glob(os.path.join(data_dir, f"{case_name}_features_frac*.npy")))
    if not feature_files:
        raise FileNotFoundError(f"No data files found for pattern: '{case_name}_features_frac*.npy' in '{data_dir}'.")
    try:
        # Extract number of buses from case name
        num_buses = int(''.join(filter(str.isdigit, case_name)))
        
        # Load adjacency matrix
        first_adj_path = feature_files[0].replace('features', 'adjacency')
        adj_object_array = np.load(first_adj_path, allow_pickle=True)
        edge_index = adj_object_array[0]
        print(f"[Data] Loaded edge index from: {os.path.basename(first_adj_path)} with shape {edge_index.shape}")
        
        static_adjacency_matrix = _convert_edge_index_to_adj(edge_index, num_buses)
        print(f"[Data] Converted to dense adjacency matrix with shape: {static_adjacency_matrix.shape}")
        
        if static_adjacency_matrix.ndim != 2 or static_adjacency_matrix.shape[0] != static_adjacency_matrix.shape[1]:
             raise ValueError(f"Conversion to dense matrix failed. Final shape is not square: {static_adjacency_matrix.shape}.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed during adjacency matrix loading and conversion: {e}")
        raise

    all_features, all_ybus, all_targets = [], [], []
    all_energy_coeffs, all_carbon_coeffs = [], []
    
    for f_path in feature_files:
        print(f"  > Loading scenario from: {os.path.basename(f_path)}")
        # --- START CORRECTION: Update filenames to match new saved data ---
        ybus_path = f_path.replace('features', 'ybus_matrices')
        # --- END CORRECTION ---
        targets_path = f_path.replace('features', 'targets')
        energy_path = f_path.replace('features', 'time_energy_coeffs').replace('.npy', '.txt')
        carbon_path = f_path.replace('features', 'time_carbon_coeffs').replace('.npy', '.txt')
        try:
            all_features.append(np.load(f_path))
            all_ybus.append(np.load(ybus_path))
            all_targets.append(np.load(targets_path))
            all_energy_coeffs.append(np.loadtxt(energy_path))
            all_carbon_coeffs.append(np.loadtxt(carbon_path))
        except FileNotFoundError as e:
            print(f"\n[CRITICAL ERROR] A required data file is missing: {e.filename}")
            print("Please ensure you have run 'gen_meas_best.py' to generate all necessary data files.")
            raise e

    # --- START CORRECTION: Concatenate all data arrays along the time axis ---
    concatenated_features = np.concatenate(all_features, axis=0)
    concatenated_ybus = np.concatenate(all_ybus, axis=0)
    concatenated_targets = np.concatenate(all_targets, axis=0)
    concatenated_energy_coeffs = np.concatenate(all_energy_coeffs, axis=0)
    concatenated_carbon_coeffs = np.concatenate(all_carbon_coeffs, axis=0)
    # --- END CORRECTION ---

    print(f"[Data] All scenarios concatenated. Total samples: {concatenated_features.shape[0]}")
    normalizer = PowerSystemNormalizer(concatenated_features)
    features_norm = normalizer.normalize(concatenated_features)
    print("[Data] Full dataset loaded and normalized.")
    
    # --- START CORRECTION: Return concatenated Ybus array ---
    return (features_norm, static_adjacency_matrix, concatenated_ybus, concatenated_targets, 
            concatenated_energy_coeffs, concatenated_carbon_coeffs, normalizer)
    # --- END CORRECTION ---

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

def create_data_loaders(features, adjacency, ybus_matrices, targets, time_energy_coeffs, time_carbon_coeffs, config, is_static):
    seq_len = 1 if is_static else getattr(config, 'SEQUENCE_LENGTH', 1)
    dataset = PowerSystemDataset(
        features, adjacency, ybus_matrices, targets, 
        time_energy_coeffs, time_carbon_coeffs, 
        is_static, seq_len
    )
    dataset_size = len(dataset)
    train_size = int(config.TRAIN_SPLIT * dataset_size)
    val_size = int(config.VAL_SPLIT * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    collate_fn_to_use = _collate_static if is_static else _collate_sequential_padded
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=collate_fn_to_use)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn_to_use)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn_to_use)
    print("[Data] DataLoaders created.")
    return train_loader, val_loader, test_loader