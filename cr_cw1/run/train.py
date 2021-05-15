import argparse
import torch
import torchvision
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

    # parse the arguments
    args = parser.parse_args()
    batch_size: int = args.batch_size
    epoch: int = args.epoch

    # gpu set up
    print("is cuda available?:", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the data
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar_train_set = torchvision.datasets.CIFAR10(root=CIFAR10_DIR, train=True,
                                                   download=False, transform=transform)
    cifar10_train_loader = torch.utils.data.DataLoader(cifar_train_set, batch_size=batch_size,
                                                       shuffle=True, num_workers=2)

    # instantiate the model
    base_cnn = BaseCNN()

    # instantiate the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(base_cnn.parameters(), lr=0.001, momentum=0.9)

    # train the network
    for epoch_idx in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0  # for logging purposes.
        for batch_idx, data in enumerate(cifar10_train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # make sure to load the data to the device
            inputs.to(device)
            labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = base_cnn(inputs)  # the logits.
            loss = criterion(outputs, labels)  # nice, handy way of computing the loss. Cross-entropy loss.
            loss.backward()  # back propagate.
            optimizer.step()  # a gradient descent step.
            # report the running loss - this is to be visualised later.
            running_loss += loss.item()
            if batch_idx % 2000 == 1999:  # print every 2000 mini-batches
                print('[epoch:%d, batch:%5d] loss: %.3f' %
                      (epoch_idx + 1, batch_idx + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    main()
