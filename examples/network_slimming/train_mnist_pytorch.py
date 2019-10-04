#!/usr/bin/env python
# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# Original code forked from PyTorch licensed pytorch project
# https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from chainerpruner.pruning import network_slimming
from chainerpruner.pruning.network_slimming.pytorch import Lasso


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        class Lambda(nn.Module):
            def __init__(self, func):
                super().__init__()
                self.func = func

            def forward(self, x):
                return self.func(x)

        self.net = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.BatchNorm2d(20),  # net.1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            nn.BatchNorm2d(50),  # net.5
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Lambda(lambda x: x.view(-1, 4 * 4 * 50)),
            nn.Linear(4 * 4 * 50, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch, lasso):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        # apply Lasso
        lasso()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    # for channel pruning
    parser.add_argument('--rate', type=float, default=0.01,
                        help='coefficient for Lasso.')
    parser.add_argument('--pruning-threshold', type=float, default=0.1,
                        help='pruning threshold for BatchNorm2d.weight (global value for entire model)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # initialize pruning extension
    # TODO(tkat0) cant pruning net.4 (conv-fc)
    target_conv_layers = ['net.0']
    target_batchnorm_layers = ['net.1']

    x = torch.randn((1, 1, 28, 28), requires_grad=False).to(device)
    lasso = Lasso(model, args.rate, target_batchnorm_layers)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, lasso)
        test(args, model, device, test_loader)

    model.to('cpu')

    if (args.save_model):
        torch.onnx.export(model, x, "mnist_cnn.onnx", verbose=False,
                          input_names=['input'],
                          output_names=['output'])

    # rebiuld model
    info = network_slimming.pruning(model, x, target_conv_layers, args.pruning_threshold)
    print(info)

    model.to(device)(x)  # testing

    if (args.save_model):
        torch.onnx.export(model, x, "mnist_cnn_rebuild.onnx", verbose=False,
                          input_names=['input'],
                          output_names=['output'])
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
