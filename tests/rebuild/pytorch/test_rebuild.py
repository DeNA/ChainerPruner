import numpy as np
from matplotlib import pyplot as plt
import pytest

from chainerpruner import Pruner, Graph
from chainerpruner.masks import NormMask
import torchvision.models as models
import torch
from torch import nn
from torch.nn import functional as F

enable_save = False


def _test_model(tmpdir, model_class, x, target_layers, percent, options=None, train=False, save=False):
    # save initialized model
    if not options:
        options = dict()
    model = model_class(**options)
    if train:
        model.train()
    else:
        model.eval()
    if save:
        PATH = str(tmpdir.join('model.onnx'))
        torch.onnx.export(model, x, PATH, verbose=False,
                          input_names=['input'],
                          output_names=['output'])

    # print(model)

    # pruning with Pruner
    graph = Graph(model, x)
    PATH = str(tmpdir.join('model.png'))
    graph.plot()
    plt.savefig(PATH)

    mask = NormMask(model, graph, target_layers, percent=percent)
    pruner = Pruner(model, x, target_layers, mask)
    pruner.apply_mask()
    info = pruner.apply_rebuild()

    model(x)

    PATH = str(tmpdir.join('model.pth'))
    torch.save(model.state_dict(), PATH)

    # load pruned weight and run
    model = model_class(**options)
    model.load_state_dict(torch.load(PATH))
    if train:
        model.train()
    else:
        model.eval()
    model(x)

    # save pruned onnx
    if save:
        PATH = str(tmpdir.join('model_pruned.onnx'))
        torch.onnx.export(model, x, PATH, verbose=False,
                          input_names=['input'],
                          output_names=['output'])


def test_resnet18(tmpdir):
    model_class = models.resnet18
    x = torch.randn((1, 3, 32, 32), requires_grad=False)
    target_layers = [
        'layer1.0.conv1',
        'layer4.1.conv1',
    ]
    percent = 0.8

    _test_model(tmpdir, model_class, x, target_layers, percent, save=enable_save)

def test_mobilenetv2(tmpdir):
    model_class = models.mobilenet_v2
    x = torch.randn((1, 3, 32, 32), requires_grad=False)
    target_layers = [
        'features.0.0',
    ]
    percent = 0.7

    _test_model(tmpdir, model_class, x, target_layers, percent, save=enable_save)


def test_no_target_layers():
    x = torch.randn((1, 3, 32, 32), requires_grad=False)
    model = models.resnet18()
    model.eval()

    model(x)

    percent = 0.8
    target_layers = []  # empty!

    graph = Graph(model, x)
    mask = NormMask(model, graph, target_layers, percent=percent)
    pruner = Pruner(model, x, target_layers, mask)
    pruner.apply_mask()
    try:
        info = pruner.apply_rebuild()
        raise ValueError
    except ValueError:
        pass

def test_block(tmpdir):

    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, ksize, stride, pad):
            super(Block, self).__init__()
            self.convDW = nn.Conv2d(in_channels, in_channels, ksize, stride=stride, padding=pad, groups=in_channels, bias=False)
            self.bnDW = nn.BatchNorm2d(in_channels)
            self.convPW = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
            self.bnPW = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            inplace = True
            # inplace = False TODO(tkat0) can't trace
            h = F.relu(self.bnDW(self.convDW(x)), inplace=inplace)
            h = F.relu(self.bnPW(self.convPW(h)), inplace=inplace)
            return h

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()

            self.conv1 = Block(3, 7, 3, 1, 0)
            self.conv2 = Block(7, 15, 3, 1, 0)

        def forward(self, x):
            h = self.conv1(x)
            h = self.conv2(h)
            return h

    model_class = SimpleNet
    x = torch.randn((1, 3, 32, 32), requires_grad=False)
    target_layers = ['conv1.convPW']
    percent = 0.7

    model_class()(x)

    _test_model(tmpdir, model_class, x, target_layers, percent, options=None, save=enable_save)


def test_upsample(tmpdir):

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 7, 1, stride=1, padding=0, bias=False)
            self.up = nn.Upsample(scale_factor=2)
            self.conv2 = nn.Conv2d(7, 15, 1, stride=1, padding=0, bias=False)

        def forward(self, x):
            h = self.conv1(x)
            h = self.up(h)
            h = self.conv2(h)
            return h

    model_class = SimpleNet
    x = torch.randn((1, 3, 32, 32), requires_grad=False)
    target_layers = ['conv1']
    percent = 0.7

    model_class()(x)

    _test_model(tmpdir, model_class, x, target_layers, percent, options=None, save=enable_save)
