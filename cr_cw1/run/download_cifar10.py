"""
code excerpted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import torchvision
from cr_cw1.paths import CIFAR10_DIR


def main():
    # download the train and test set.
    torchvision.datasets.CIFAR10(root=CIFAR10_DIR, train=True, download=True)
    torchvision.datasets.CIFAR10(root=CIFAR10_DIR, train=False, download=True)


if __name__ == '__main__':
    main()
