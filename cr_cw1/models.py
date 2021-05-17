"""
the model architecture was adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Model(pl.LightningModule):
    """
    the training step is defined here.
    """

    def __init__(self, lr: float, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        training step defines the train loop.
        """
        x, y = batch
        y_hat = self.forward(x)
        # we compute the
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss:", loss)
        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        validation step defines the validation loop.
        # the batches will come from the validation dataset.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        # log the validation loss
        self.log("val_loss", loss)

    def configure_optimizers(self):
        # we use an adam optimizer for this model.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class BaseCNN(Model):
    """
    the baseline CNN.
    Just a single Conv2d layer.
    One fully connected layer.
    """
    def __init__(self, lr: float):
        super(BaseCNN, self).__init__(lr)
        # input channel 3 = the R, G and B channels.
        # change this with a sequential layer.
        self.layer_1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            torch.nn.ReLU(),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # sequentially define the layer.
        self.fc_1 = nn.Linear(in_features=32 * 15 * 15, out_features=10)  # should match.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        in: a 2d image tensor
        out: a 1d logit vector. 10 logits.
        """
        out = self.layer_1(x)
        out = out.view(out.size(0), -1)  # flatten for the fully connected layer
        out = self.fc_1(out)
        return out

# what we should first do, is optimising the epoch with the base CNN. That's the first thing you must optimise, alright?


class TwoCNN(Model):
    """
    the baseline CNN.
    Just a single Conv2d layer.
    One fully connected layer.
    """
    def __init__(self, lr: float):
        super(TwoCNN, self).__init__(lr)
        # input channel 3 = the R, G and B channels.
        # change this with a sequential layer.
        self.layer_1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            torch.nn.ReLU(),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # sequentially define the layer.
        self.layer_2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            torch.nn.ReLU(),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # sequentially define the layer.
        self.fc_1 = nn.Linear(in_features=32 * 6 * 6, out_features=10)  # should match.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        in: a 2d image tensor
        out: a 1d logit vector. 10 logits.
        """
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = out.view(out.size(0), -1)  # flatten for the fully connected layer
        out = self.fc_1(out)
        return out


class ThreeCNN(Model):
    """
    the baseline CNN.
    Just a single Conv2d layer.
    One fully connected layer.
    """
    def __init__(self, lr: float):
        super(ThreeCNN, self).__init__(lr)
        # input channel 3 = the R, G and B channels.
        # change this with a sequential layer.
        self.layer_1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            torch.nn.ReLU(),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # sequentially define the layer.
        self.layer_2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            torch.nn.ReLU(),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # sequentially define the layer.
        self.layer_3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            torch.nn.ReLU(),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # sequentially define the layer.
        self.fc_1 = nn.Linear(in_features=32 * 2 * 2, out_features=10)  # should match.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        in: a 2d image tensor
        out: a 1d logit vector. 10 logits.
        """
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = out.view(out.size(0), -1)  # flatten for the fully connected layer
        out = self.fc_1(out)
        return out

