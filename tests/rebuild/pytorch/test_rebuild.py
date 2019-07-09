import numpy as np
from matplotlib import pyplot as plt
import pytest

from chainerpruner import Pruner, Graph
from chainerpruner.masks import NormMask
import torchvision.models as models
import torch

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
    # print(model)

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
