import lightning as L
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class StandardGCN(L.LightningModule):
    """
    Model 1: Standard GCN
    Uses a fixed static adjacency matrix for message passing.
    Spatial-only model — processes a single timestep at a time.
    Input: (num_nodes, num_features) with a fixed edge_index.
    Output: (num_nodes, out_channels)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.patience = kwargs.get('lr_patience', 10)
        self.factor = kwargs.get('lr_factor', 0.5)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.output_layer = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.loss_fn = nn.MSELoss()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x = self.relu(conv(x, edge_index, edge_weight))
        return self.output_layer(x)

    def _shared_step(self, batch, stage):
        x, edge_index, targets = batch["features"], batch["edge_index"], batch["targets"]
        preds = self(x, edge_index)
        loss = self.loss_fn(preds, targets)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=x.size(0))
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
