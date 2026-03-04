import lightning as L
import torch
from torch import nn
from torch_geometric.nn import GCNConv

from src.models.physics_loss import PhysicsLoss


class PIGCN(L.LightningModule):
    """
    Model 3: PIGCN (Physics-Informed GCN)
    Adaptive GCN that uses dynamic topology — the adjacency matrix
    changes per timestep based on contingency state.
    Physics-informed: power balance + voltage limit + branch capacity losses.
    Spatial-only model.
    Input: (num_nodes, num_features) with dynamic edge_index.
    Output: (num_nodes, out_channels)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.patience = kwargs.get('lr_patience', 10)
        self.factor = kwargs.get('lr_factor', 0.5)

        # Physics loss weights (configurable from training.yaml)
        self.lambda_power = kwargs.get('lambda_power_balance', 0.1)
        self.lambda_voltage = kwargs.get('lambda_voltage_limit', 0.01)
        self.lambda_branch = kwargs.get('lambda_branch_capacity', 0.01)

        # Physics loss module (initialized lazily from first batch)
        self._physics_loss = None

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.output_layer = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.data_loss_fn = nn.MSELoss()

    def _get_physics_loss(self, batch):
        """Lazily initialize PhysicsLoss from batch tensors (moves to correct device)."""
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

    def forward(self, x, edge_index):
        out = x
        for conv in self.convs:
            out = self.relu(conv(out, edge_index))
        return self.output_layer(out)

    def _shared_step(self, batch, stage):
        x, edge_index = batch["features"], batch["edge_index"]
        full_targets = batch["targets"]
        # Prediction targets: VM (index 8) and VA (index 9)
        targets_vm_va = full_targets[..., 8:10]

        preds = self(x, edge_index)
        data_loss = self.data_loss_fn(preds, targets_vm_va)

        # Physics-informed loss
        physics = self._get_physics_loss(batch)
        vm_pred = preds[..., 0]  # Predicted VM deviation
        va_pred = preds[..., 1]  # Predicted VA radians
        physics_result = physics(vm_pred, va_pred, full_targets)

        total_loss = data_loss + physics_result["physics_loss"]

        # Logging
        self.log(f"{stage}_loss", total_loss, prog_bar=True, batch_size=x.size(0))
        self.log(f"{stage}_data_loss", data_loss, batch_size=x.size(0))
        self.log(f"{stage}_power_balance", physics_result["power_balance_loss"], batch_size=x.size(0))
        self.log(f"{stage}_voltage_limit", physics_result["voltage_limit_loss"], batch_size=x.size(0))
        self.log(f"{stage}_branch_capacity", physics_result["branch_capacity_loss"], batch_size=x.size(0))
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
