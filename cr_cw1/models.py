"""
the model is from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    """
    the baseline CNN.
    Just a single Conv2d layer.
    One fully connected layer.
    """
    def __init__(self):
        super(BaseCNN, self).__init__()
        # input channel 3 = the R, G and B channels.
        # change this with a sequential layer.
        self.layer_1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3)),
            torch.nn.ReLU(),  # non-linear activation
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # sequentially define the layer.
        self.fc_1 = nn.Linear(in_features=6 * 15 * 15, out_features=10)  # should match.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        in: a 2d image tensor
        out: a 1d logit vector. 10 logits.
        """
        out = self.layer_1(x)
        out = out.view(out.size(0), -1)  # flatten for the fully connected layer
        out = self.fc_1(out)
        return out
