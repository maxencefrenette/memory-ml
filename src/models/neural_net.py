import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import BinaryCalibrationError


class NNHiddenLayer(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layer(x)


class NNMemoryModel(L.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        reviews_history_size: int,
        hidden_layer_size: int,
        hidden_layer_num: int,
        dropout: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(2 + 4 * reviews_history_size, hidden_layer_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            *[
                NNHiddenLayer(hidden_layer_size, dropout)
                for _ in range(hidden_layer_num)
            ],
            nn.Linear(hidden_layer_size, 1),
            nn.Sigmoid(),
        )

        self.loss_fn = nn.BCELoss()
        self.calibration_fn = BinaryCalibrationError(n_bins=20, norm="l2")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        self.log("val/loss", self.loss_fn(pred, y))
        self.log("val/calibration", self.calibration_fn(pred, y))

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        self.log("test/loss", self.loss_fn(pred, y))
        self.log("test/calibration", self.calibration_fn(pred, y))

    def configure_optimizers(self):
        return self.optimizer
