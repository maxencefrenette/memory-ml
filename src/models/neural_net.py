import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import BinaryCalibrationError


class NeuralNetwork(nn.Module):
    def __init__(self, reviews_history_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2 + 4 * reviews_history_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class NNMemoryModel(L.LightningModule):
    def __init__(self, learning_rate: float, reviews_history_size: int):
        super().__init__()
        self.save_hyperparameters()

        self.model = NeuralNetwork(
            reviews_history_size=self.hparams.reviews_history_size
        )

        self.loss_fn = nn.BCELoss()
        self.calibration_fn = BinaryCalibrationError(n_bins=20, norm="l2")

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
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
