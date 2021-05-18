"""
the model architecture was adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CNN(pl.LightningModule):
    """
    the training step is defined here.
    """

    def __init__(self, lr: float, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # should be implemented by subclasses
        raise NotImplementedError

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


class BaseCNN(CNN):
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
            torch.nn.ReLU(inplace=True),  # non-linear activation
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


class RegBaseCNN(BaseCNN):
    """
    the base CNN, but regularised with batch norm & dropout layers.
    """

    def __init__(self, lr: float):
        super().__init__(lr)
        # input channel 3 = the R, G and B channels.
        # change this with a sequential layer.
        self.layer_1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05)
        )  # sequentially define the layer.
        self.fc_1 = nn.Linear(in_features=32 * 15 * 15, out_features=10)  # should match.


class TwoCNN(CNN):
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
            torch.nn.ReLU(inplace=True),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # sequentially define the layer.
        self.layer_2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            torch.nn.ReLU(inplace=True),  # non-linear activation
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


class RegTwoCNN(TwoCNN):

    def __init__(self, lr: float):
        super().__init__(lr)
        # input channel 3 = the R, G and B channels.
        # change this with a sequential layer.
        self.layer_1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05)
        )  # sequentially define the layer.
        self.layer_2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05)
        )  # sequentially define the layer.
        self.fc_1 = nn.Linear(in_features=32 * 6 * 6, out_features=10)  # should match.


class ThreeCNN(CNN):
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
            torch.nn.Conv2d(in_channels=3,
                            out_channels=32,
                            kernel_size=(3, 3)),  # 2d convolution
            torch.nn.ReLU(inplace=True),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # sequentially define the layer.
        self.layer_2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=32,
                            kernel_size=(3, 3)),  # 2d convolution
            torch.nn.ReLU(inplace=True),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # sequentially define the layer.
        self.layer_3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32,
                            out_channels=32, kernel_size=(3, 3)),
            torch.nn.ReLU(inplace=True),  # non-linear activation
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


class RegThreeCNN(ThreeCNN):
    """
    the CNN with three convolution layers
    and one fully connected layer. Each convolution
    layer is regularised with batch norm and dropout layers.
    """
    def __init__(self, lr: float):
        super(RegThreeCNN, self).__init__(lr)
        # input channel 3 = the R, G and B channels.
        # change this with a sequential layer.
        self.layer_1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(32),  # batch norm layer
            torch.nn.ReLU(inplace=True),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.05)  # drop out layer
        )  # sequentially define the layer.
        self.layer_2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),  # batch norm layer
            torch.nn.ReLU(inplace=True),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.05) # dropout layer
        )  # sequentially define the layer.
        self.layer_3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),  # batch norm layer
            torch.nn.ReLU(inplace=True),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.05)  # dropout layer
        )  # sequentially define the layer.
        self.fc_1 = nn.Linear(in_features=32 * 2 * 2, out_features=10)  # should match.
