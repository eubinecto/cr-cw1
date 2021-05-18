import argparse
import torch
import torchvision
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from cr_cw1.paths import *
from cr_cw1.models import BaseCNN, TwoCNN, ThreeCNN, RegBaseCNN, RegTwoCNN, RegThreeCNN
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers


def eval_acc(model: torch.nn.Module, data_loader: DataLoader) -> float:
    total = 0
    correct = 0
    model.eval()  # first, put it into eval mode.
    with torch.no_grad():  # no need for calculating the gradients
        for data in data_loader:
            x, y = data
            y_hat = model.forward(x)
            _, predicted = torch.max(y_hat.data, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * (correct / total)


def main():
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int,
                        default=20)
    parser.add_argument("--epoch", type=int,
                        default=1)
    parser.add_argument("--model_name", type=str,
                        default="base_cnn")
    parser.add_argument("--lr", type=float,
                        default=1e-3)
    parser.add_argument("--patience", type=int,
                        default=3)
    parser.add_argument('--normalise_data',
                        dest='normalise_data',
                        default=False,
                        action='store_true')

    # --- get the hyper parameters --- #
    args = parser.parse_args()
    batch_size: int = args.batch_size
    epoch: int = args.epoch
    model_name: str = args.model_name
    lr: float = args.lr
    patience: int = args.patience
    normalise_data: bool = args.normalise_data

    # --- instantiate the model --- #
    if model_name == "base_cnn":
        model = BaseCNN(lr)
        default_root_dir = BASE_CNN_DIR
    elif model_name == "two_cnn":
        model = TwoCNN(lr)
        default_root_dir = TWO_CNN_DIR
    elif model_name == "three_cnn":
        model = ThreeCNN(lr)
        default_root_dir = THREE_CNN_DIR
    elif model_name == "reg_base_cnn":
        model = RegBaseCNN(lr)
        default_root_dir = REG_BASE_CNN_DIR
    elif model_name == "reg_two_cnn":
        model = RegTwoCNN(lr)
        default_root_dir = REG_TWO_CNN_DIR
    elif model_name == "reg_three_cnn":
        model = RegThreeCNN(lr)
        default_root_dir = REG_THREE_CNN_DIR
    else:
        raise ValueError("invalid model name:", model_name)

    # --- gpu set up --- #
    print("is cuda available?:", torch.cuda.is_available())

    # --- normalise or not --- #
    if normalise_data:
        # The output of torchvision datasets are PILImage images of range [0, 1].
        # We transform them to Tensors of normalized range [-1, 1].
        pipeline = [
            transforms.ToTensor(),  # transform PILImage to pytorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise the values
        ]
    else:
        pipeline = [
            transforms.ToTensor()
        ]

    # --- load the data --- #
    transform = transforms.Compose(pipeline)
    train_set = torchvision.datasets.CIFAR10(root=CIFAR10_DIR, train=True,
                                             download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=CIFAR10_DIR, train=False,
                                            download=False, transform=transform)  # to be used for eval
    # do the train / val split. (80 to 20)
    train_len = int(0.8 * len(train_set))
    val_len = len(train_set) - train_len
    train_set, val_set = random_split(train_set, [train_len, val_len])

    # --- instantiate the loaders --- #
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=4)

    # --- instantiate early stopping --- #
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=True,  # so that I can plot the trend.
        mode='min'  # look for the minimum.
    )

    # --- logger setup --- #
    tb_logger = pl_loggers.CSVLogger(save_dir=default_root_dir)

    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else None,
                         max_epochs=epoch,
                         log_every_n_steps=1,
                         default_root_dir=default_root_dir,
                         logger=tb_logger,
                         # this will do the early stopping for you.
                         callbacks=[early_stopping])

    # --- start training --- #
    trainer.fit(model=model,
                train_dataloader=train_loader,
                val_dataloaders=val_loader)

    # --- evaluate the training & testing accuracies of the model --- #
    train_acc = eval_acc(model, train_loader)
    test_acc = eval_acc(model, test_loader)
    print('Accuracy of the network on the train images:', train_acc)
    print('Accuracy of the network on the test images:', test_acc)


if __name__ == '__main__':
    main()
