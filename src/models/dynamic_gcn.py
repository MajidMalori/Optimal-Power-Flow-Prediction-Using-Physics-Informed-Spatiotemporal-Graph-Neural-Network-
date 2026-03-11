import lightning as L
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class DynamicGCN(L.LightningModule):
    """
    Model 2: Dynamic GCN
    Uses the real-time post-contingency adjacency matrix for message passing.
    When a line trips, message flow between disconnected buses stops immediately.
    Spatial-only model — processes a single timestep at a time.
    Input: (num_nodes, num_features) with a dynamic edge_index that changes per timestep.
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

    def forward(self, x, edge_indices, edge_weight=None):
        """
        Args:
            x: [B, N, F]
            edge_indices: List of length B, each [2, E_i]
        """
        batch_size = x.size(0)
        outputs = []
        
        for b in range(batch_size):
            # Process each sample with its own topology
            sample_x = x[b]
            sample_edge_index = edge_indices[b]
            
            out = sample_x
            for conv in self.convs:
                out = self.relu(conv(out, sample_edge_index, edge_weight))
            outputs.append(self.output_layer(out))
            
        return torch.stack(outputs)

    def _get_physics_loss(self, batch):
        """Lazily initialize PhysicsLoss for evaluation (even if not used for training)."""
        if not hasattr(self, '_physics_loss') or self._physics_loss is None:
            from src.models.physics_loss import PhysicsLoss
            self._physics_loss = PhysicsLoss(
                ybus=batch["ybus"].to(self.device),
                branch_from=batch["branch_from"].to(self.device),
                branch_to=batch["branch_to"].to(self.device),
                branch_max_s_pu=batch["branch_max_s_pu"].to(self.device),
                contingencies=batch.get("contingencies", torch.tensor([])).to(self.device),
            )
        return self._physics_loss

    def _shared_step(self, batch, stage):
        x, edge_indices = batch["features"], batch["edge_index"]
        # Slice targets to VM=8, VA=9 for data loss (baseline — no physics)
        targets = batch["targets"][..., 8:10]
        preds = self(x, edge_indices)
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
