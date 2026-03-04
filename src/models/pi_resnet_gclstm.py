import lightning as L
import torch
from torch import nn
from .layers import ResidualGCNBlock


class PIResnetGCLSTM(L.LightningModule):
    """
    Model 6: PIResnetGCLSTM (Physics-Informed ResNet GCN + LSTM)
    Uses Residual GCN blocks to build deeper spatial networks without oversmoothing,
    then processes the temporal sequence through an LSTM.
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

        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(ResidualGCNBlock(in_channels, gcn_hidden))
        for _ in range(num_res_blocks - 1):
            self.res_blocks.append(ResidualGCNBlock(gcn_hidden, gcn_hidden))

        self.lstm = nn.LSTM(input_size=gcn_hidden, hidden_size=lstm_hidden, batch_first=True)
        self.output_layer = nn.Linear(lstm_hidden, out_channels)
        self.loss_fn = nn.MSELoss()

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
            for res_block in self.res_blocks:
                out_t = res_block(out_t, edge_index_t)
            spatial_embeddings.append(out_t)

        spatial_seq = torch.stack(spatial_embeddings, dim=1)
        lstm_out, _ = self.lstm(spatial_seq)
        last_out = lstm_out[:, -1, :]
        preds = self.output_layer(last_out)
        return preds.reshape(batch_size, num_nodes, -1)

    def _shared_step(self, batch, stage):
        x_seq = batch["features"]
        edge_idx_seq = batch["edge_index_seq"]
        targets = batch["targets"]
        preds = self(x_seq, edge_idx_seq)
        loss = self.loss_fn(preds, targets)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=x_seq.size(0))
        return loss

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
