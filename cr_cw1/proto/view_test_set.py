import torchvision
from torchvision import transforms

from cr_cw1.paths import CIFAR10_DIR


def main():
    transform = transforms.Compose([
            transforms.ToTensor(),  # transform PILImage to pytorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise the values
         ]
    )
    test_set = torchvision.datasets.CIFAR10(root=CIFAR10_DIR, train=False,
                                            download=False, transform=transform)

    for x, y in test_set:
        print(x)
        print(y)


if __name__ == '__main__':
    main()
