import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torchsample.modules import ModuleTrainer
from torchvision.transforms import transforms
from cr_cw1.paths import CIFAR10_DIR
from cr_cw1.models import BaseCNN


def main():
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int,
                        default=5)
    parser.add_argument("--epoch", type=int,
                        default=10)
    parser.add_argument("--model_name", type=str,
                        default="BaseCNN")

    # parse the arguments
    args = parser.parse_args()
    batch_size: int = args.batch_size
    epoch: int = args.epoch
    model_name: str = args.model_name

    # --- instantiate the model --- #
    if model_name == "BaseCNN":
        model = BaseCNN()
    else:
        raise ValueError("invalid model name:", model_name)

    # --- gpu set up --- #
    print("is cuda available?:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- load the data --- #
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    transform = transforms.Compose([
            transforms.ToTensor(),  # transform PILImage to pytorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise the values
         ]
    )
    train_set = torchvision.datasets.CIFAR10(root=CIFAR10_DIR, train=True,
                                             download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=CIFAR10_DIR, train=False,
                                            download=False, transform=transform)

    # --- instantiate the loaders --- #
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    # known issue: https://github.com/ncullen93/torchsample/issues/65#issuecomment-353924500
    # here, I just register them as 1.
    train_loader.dataset.num_inputs = 1
    train_loader.dataset.num_targets = 1
    test_loader.dataset.num_inputs = 1
    test_loader.dataset.num_targets = 1

    # --- instantiate the trainer --- #
    trainer = ModuleTrainer(model)
    trainer.compile(loss='nll_loss',
                    optimizer='adadelta')

    # --- fit the model --- #
    trainer.fit_loader(loader=train_loader,
                       val_loader=test_loader,
                       num_epoch=epoch,
                       cuda_device=1 if torch.cuda.is_available() else -1,
                       verbose=1)


if __name__ == '__main__':
    main()
