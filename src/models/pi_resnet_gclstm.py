import lightning as L
import torch
from torch import nn
from .layers import ResidualGCNBlock

from src.models.physics_loss import PhysicsLoss


class PIResnetGCLSTM(L.LightningModule):
    """
    Model 6: PIResnetGCLSTM (Physics-Informed ResNet GCN + LSTM)
    Uses Residual GCN blocks to build deeper spatial networks without oversmoothing,
    then processes the temporal sequence through an LSTM.
    Physics-informed: power balance + voltage limit + branch capacity losses.
    Input: (batch, seq_len, num_nodes, num_features) with edge_index per step.
    Output: (batch, num_nodes, out_channels) — prediction for the last timestep.
    """

    def __init__(self, in_channels, gcn_hidden, lstm_hidden, out_channels,
                 num_res_blocks=3, **kwargs):
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

        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(ResidualGCNBlock(in_channels, gcn_hidden))
        for _ in range(num_res_blocks - 1):
            self.res_blocks.append(ResidualGCNBlock(gcn_hidden, gcn_hidden))

        self.lstm = nn.LSTM(input_size=gcn_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.output_layer = nn.Linear(lstm_hidden, out_channels)
        self.data_loss_fn = nn.MSELoss()

    def _get_physics_loss(self, batch):
        if self._physics_loss is None:
            self._physics_loss = PhysicsLoss(
                ybus=batch["ybus"].to(self.device),
                branch_from=batch["branch_from"].to(self.device),
                branch_to=batch["branch_to"].to(self.device),
                branch_max_s_pu=batch["branch_max_s_pu"].to(self.device),
                contingencies=batch.get("contingencies", torch.tensor([])).to(self.device),
                lambda_power=self.lambda_power,
                lambda_voltage=self.lambda_voltage,
                lambda_branch=self.lambda_branch,
            )
        return self._physics_loss

    def forward(self, x_seq, edge_idx_seq):
        """
        x_seq: (batch, seq_len, nodes, features)
        edge_idx_seq: list (batch) of lists (seq) of edge_index tensors
        """
        batch_size, seq_len, num_nodes, num_features = x_seq.shape

        spatial_embeddings = []
        for b in range(batch_size):
            sample_embeddings = []
            for t in range(seq_len):
                x_bt = x_seq[b, t]
                edge_index_bt = edge_idx_seq[b][t]
                
                out_t = x_bt
                for res_block in self.res_blocks:
                    out_t = res_block(out_t, edge_index_bt)
                sample_embeddings.append(out_t)
            spatial_embeddings.append(torch.stack(sample_embeddings))

        spatial_seq = torch.stack(spatial_embeddings)
        # Reshape for LSTM: (batch * nodes, seq_len, hidden)
        spatial_seq = spatial_seq.transpose(1, 2).reshape(batch_size * num_nodes, seq_len, -1)
        
        lstm_out, _ = self.lstm(spatial_seq)
        last_out = lstm_out[:, -1, :]
        preds = self.output_layer(last_out)
        return preds.reshape(batch_size, num_nodes, -1)

    def _shared_step(self, batch, stage):
        x_seq = batch["features"]
        edge_idx_seq = batch["edge_index_seq"]
        topology_ids_seq = batch["topology_ids"]
        last_topo_ids = topology_ids_seq[:, -1]
        
        full_targets = batch["targets"]
        targets_vm_va = full_targets[..., 8:10]

        preds = self(x_seq, edge_idx_seq)
        data_loss = self.data_loss_fn(preds, targets_vm_va)

        # Physics-informed loss
        physics = self._get_physics_loss(batch)
        vm_pred = preds[..., 0]
        va_pred = preds[..., 1]
        physics_result = physics(vm_pred, va_pred, full_targets, last_topo_ids)

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
