import lightning as L
import torch
from torch import nn
from torch_geometric.nn import GCNConv

from src.models.physics_loss import PhysicsLoss


class PIGCGRU(L.LightningModule):
    """
    Model 5: PIGCGRU (Physics-Informed GCN + GRU)
    Faster temporal processing than LSTM using Gated Recurrent Units.
    Spatial features extracted by GCN with dynamic adjacency,
    passed sequentially through a GRU.
    Physics-informed: power balance + voltage limit + branch capacity losses.
    Input: (batch, seq_len, num_nodes, num_features) with edge_index per step.
    Output: (batch, num_nodes, out_channels) — prediction for the last timestep.
    """

    def __init__(self, in_channels, gcn_hidden, gru_hidden, out_channels,
                 num_gcn_layers=2, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.patience = kwargs.get('lr_patience', 10)
        self.factor = kwargs.get('lr_factor', 0.5)

        # Physics loss weights
        self.lambda_power = kwargs.get('lambda_power_balance', 0.1)
        self.lambda_voltage = kwargs.get('lambda_voltage_limit', 0.01)
        self.lambda_branch = kwargs.get('lambda_branch_capacity', 0.01)
        self._physics_loss = None

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, gcn_hidden))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(gcn_hidden, gcn_hidden))

        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden, batch_first=True)
        self.output_layer = nn.Linear(gru_hidden, out_channels)
        self.data_loss_fn = nn.MSELoss()

    def _get_physics_loss(self, batch):
        if self._physics_loss is None:
            self._physics_loss = PhysicsLoss(
                ybus=batch["ybus"].to(self.device),
                branch_from=batch["branch_from"].to(self.device),
                branch_to=batch["branch_to"].to(self.device),
                branch_max_s_pu=batch["branch_max_s_pu"].to(self.device),
                lambda_power=self.lambda_power,
                lambda_voltage=self.lambda_voltage,
                lambda_branch=self.lambda_branch,
            )
        return self._physics_loss

    def forward(self, x_seq, edge_idx_seq):
        """
        x_seq: (batch, seq_len, nodes, features)
        edge_idx_seq: list of edge_index tensors, one per timestep
        """
        batch_size, seq_len, num_nodes, num_features = x_seq.shape

        spatial_embeddings = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :, :].reshape(-1, num_features)
            edge_index_t = edge_idx_seq[t]

            out_t = x_t
            for gcn in self.gcn_layers:
                out_t = self.relu(gcn(out_t, edge_index_t))
            spatial_embeddings.append(out_t)

        spatial_seq = torch.stack(spatial_embeddings, dim=1)
        gru_out, _ = self.gru(spatial_seq)
        last_out = gru_out[:, -1, :]
        preds = self.output_layer(last_out)
        return preds.reshape(batch_size, num_nodes, -1)

    def _shared_step(self, batch, stage):
        x_seq = batch["features"]
        edge_idx_seq = batch["edge_index_seq"]
        full_targets = batch["targets"]
        targets_vm_va = full_targets[..., 8:10]

        preds = self(x_seq, edge_idx_seq)
        data_loss = self.data_loss_fn(preds, targets_vm_va)

        # Physics-informed loss
        physics = self._get_physics_loss(batch)
        vm_pred = preds[..., 0]
        va_pred = preds[..., 1]
        physics_result = physics(vm_pred, va_pred, full_targets)

        total_loss = data_loss + physics_result["physics_loss"]

        self.log(f"{stage}_loss", total_loss, prog_bar=True, batch_size=x_seq.size(0))
        self.log(f"{stage}_data_loss", data_loss, batch_size=x_seq.size(0))
        self.log(f"{stage}_power_balance", physics_result["power_balance_loss"], batch_size=x_seq.size(0))
        self.log(f"{stage}_voltage_limit", physics_result["voltage_limit_loss"], batch_size=x_seq.size(0))
        self.log(f"{stage}_branch_capacity", physics_result["branch_capacity_loss"], batch_size=x_seq.size(0))
        return total_loss

    def training_step(self, batch, _batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, _batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience, factor=self.factor
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
