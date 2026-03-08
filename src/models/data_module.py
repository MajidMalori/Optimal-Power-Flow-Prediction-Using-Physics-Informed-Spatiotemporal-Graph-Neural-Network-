import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


class PowerFlowDataset(Dataset):
    """
    Dataset for spatial-only models (StandardGCN, DynamicGCN, PIGCN).
    Returns individual timesteps.
    """
    def __init__(self, features, targets, topology_ids, ybus_base, contingencies, cont_timesteps, adj):
        self.features = features
        self.targets = targets
        self.topology_ids = topology_ids
        self.ybus_base = ybus_base
        self.contingencies = contingencies
        self.cont_timesteps = cont_timesteps
        self.adj = adj

    def __len__(self):
        return len(self.features)

    def _get_edge_index(self, idx):
        topo_id = self.topology_ids[idx].item()
        if topo_id == 0 or len(self.contingencies) == 0:
            return self.adj
        else:
            # Reconstruct adjacency from contingency Ybus
            ybus = self.contingencies[topo_id - 1]
            ybus_abs = torch.abs(ybus)
            ybus_abs.fill_diagonal_(0)
            edge_index = (ybus_abs > 1e-6).nonzero().t().contiguous()
            return edge_index

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "targets": self.targets[idx],
            "topology_id": self.topology_ids[idx],
            "edge_index": self._get_edge_index(idx)
        }


class SpatioTemporalDataset(Dataset):
    """
    Dataset for spatio-temporal models (LSTM/GRU).
    Returns sequences of length `seq_len`.
    """
    def __init__(self, features, targets, topology_ids, ybus_base, contingencies, cont_timesteps, adj, seq_len):
        self.features = features
        self.targets = targets
        self.topology_ids = topology_ids
        self.ybus_base = ybus_base
        self.contingencies = contingencies
        self.cont_timesteps = cont_timesteps
        self.adj = adj
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def _get_edge_index(self, idx):
        topo_id = self.topology_ids[idx].item()
        if topo_id == 0 or len(self.contingencies) == 0:
            return self.adj
        else:
            ybus = self.contingencies[topo_id - 1]
            ybus_abs = torch.abs(ybus)
            ybus_abs.fill_diagonal_(0)
            edge_index = (ybus_abs > 1e-6).nonzero().t().contiguous()
            return edge_index

    def __getitem__(self, idx):
        feat_seq = self.features[idx : idx + self.seq_len]
        # Target is the prediction for the end of the sequence
        # targets shape: (T, num_nodes, num_features)
        target = self.targets[idx + self.seq_len - 1]
        
        topo_ids_seq = self.topology_ids[idx : idx + self.seq_len]
        edge_index_seq = [self._get_edge_index(i) for i in range(idx, idx + self.seq_len)]
        
        return {
            "features": feat_seq,
            "targets": target,
            "topology_ids": topo_ids_seq,
            "edge_index_seq": edge_index_seq
        }


class PowerFlowDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, case_name: str, batch_size: int = 32, seq_len: int = 1):
        super().__init__()
        self.data_dir = os.path.join(data_dir, case_name)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.meta = {}

    def setup(self, stage=None):
        with open(os.path.join(self.data_dir, 'normalization.json')) as f:
            self.meta = json.load(f)

        self.ybus_base = torch.load(os.path.join(self.data_dir, 'ybus_base.pt'), weights_only=True)
        self.adj = torch.load(os.path.join(self.data_dir, 'adjacency.pt'), weights_only=True)
        
        cont_path = os.path.join(self.data_dir, 'ybus_contingencies.pt')
        self.contingencies = torch.load(cont_path, weights_only=True) if os.path.exists(cont_path) else torch.tensor([])
        
        ts_path = os.path.join(self.data_dir, 'ybus_contingency_timesteps.pt')
        self.cont_timesteps = torch.load(ts_path, weights_only=True) if os.path.exists(ts_path) else torch.tensor([])

        # Load branch constraint data for physics-informed loss
        bf_path = os.path.join(self.data_dir, 'branch_from.pt')
        bt_path = os.path.join(self.data_dir, 'branch_to.pt')
        bm_path = os.path.join(self.data_dir, 'branch_max_s_pu.pt')
        self.branch_from = torch.load(bf_path, weights_only=True) if os.path.exists(bf_path) else torch.tensor([], dtype=torch.int64)
        self.branch_to = torch.load(bt_path, weights_only=True) if os.path.exists(bt_path) else torch.tensor([], dtype=torch.int64)
        self.branch_max_s_pu = torch.load(bm_path, weights_only=True) if os.path.exists(bm_path) else torch.tensor([], dtype=torch.float32)

        if stage == 'fit' or stage is None:
            train_f = torch.load(os.path.join(self.data_dir, 'train_features.pt'), weights_only=True)
            train_t = torch.load(os.path.join(self.data_dir, 'train_targets.pt'), weights_only=True)
            train_topo = torch.load(os.path.join(self.data_dir, 'train_topology_ids.pt'), weights_only=True)
            
            val_f = torch.load(os.path.join(self.data_dir, 'val_features.pt'), weights_only=True)
            val_t = torch.load(os.path.join(self.data_dir, 'val_targets.pt'), weights_only=True)
            val_topo = torch.load(os.path.join(self.data_dir, 'val_topology_ids.pt'), weights_only=True)

            if self.seq_len > 1:
                self.train_dataset = SpatioTemporalDataset(train_f, train_t, train_topo, self.ybus_base, self.contingencies, self.cont_timesteps, self.adj, self.seq_len)
                self.val_dataset = SpatioTemporalDataset(val_f, val_t, val_topo, self.ybus_base, self.contingencies, self.cont_timesteps, self.adj, self.seq_len)
            else:
                self.train_dataset = PowerFlowDataset(train_f, train_t, train_topo, self.ybus_base, self.contingencies, self.cont_timesteps, self.adj)
                self.val_dataset = PowerFlowDataset(val_f, val_t, val_topo, self.ybus_base, self.contingencies, self.cont_timesteps, self.adj)

        if stage == 'test' or stage is None:
            test_f = torch.load(os.path.join(self.data_dir, 'test_features.pt'), weights_only=True)
            test_t = torch.load(os.path.join(self.data_dir, 'test_targets.pt'), weights_only=True)
            test_topo = torch.load(os.path.join(self.data_dir, 'test_topology_ids.pt'), weights_only=True)
            
            if self.seq_len > 1:
                self.test_dataset = SpatioTemporalDataset(test_f, test_t, test_topo, self.ybus_base, self.contingencies, self.cont_timesteps, self.adj, self.seq_len)
            else:
                self.test_dataset = PowerFlowDataset(test_f, test_t, test_topo, self.ybus_base, self.contingencies, self.cont_timesteps, self.adj)

    def collate_fn(self, batch):
        """Custom collate function to handle variable-sized edge indices."""
        features = torch.stack([item["features"] for item in batch])
        full_targets = torch.stack([item["targets"] for item in batch])
        
        result = {
            "features": features,
            "targets": full_targets,
            "ybus": self.ybus_base,
            "contingencies": self.contingencies,
            "branch_from": self.branch_from,
            "branch_to": self.branch_to,
            "branch_max_s_pu": self.branch_max_s_pu,
        }
        
        if self.seq_len > 1:
            result["topology_ids"] = torch.stack([item["topology_ids"] for item in batch])
            result["edge_index_seq"] = [item["edge_index_seq"] for item in batch]
        else:
            result["topology_ids"] = torch.stack([item["topology_id"] for item in batch])
            result["edge_index"] = [item["edge_index"] for item in batch]
        
        return result

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
