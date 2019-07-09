import numpy as np
import pytest
import chainer
from chainer import links as L
from chainer import functions as F
from chainercv.links import ResNet50

from chainerpruner.rebuild.rebuild import rebuild
from chainerpruner.rebuild.calc_pruning_connection import calc_pruning_connection
from chainerpruner import Pruner, Graph
from chainerpruner.masks import NormMask
from tests.testutils import AllSupportedLayersNet


class SimpleNet(chainer.Chain):

    def __init__(self):
        super(SimpleNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 3, 3)
            self.bn1_1 = L.BatchNormalization(3)
            self.bn1_2 = L.BatchNormalization(3)
            self.bn1_3 = L.BatchNormalization(3)
            self.conv2 = L.Convolution2D(4, 3)
            self.bn2 = L.BatchNormalization(4)
            self.fc = L.Linear(10)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn1_1(h)
        h = self.bn1_2(h)
        h = self.bn1_3(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = F.average_pooling_2d(h, ksize=h.shape[2:])  # GAP
        h = self.fc(h)
        return h


def test_allnet():
    x = np.zeros((1, 3, 32, 32), dtype=np.float32)
    model = AllSupportedLayersNet()

    model(x)

    percent = 0.8
    target_layers = AllSupportedLayersNet.target_layers

    graph = Graph(model, x)
    mask = NormMask(model, graph, target_layers, percent=percent)
    pruner = Pruner(model, x, target_layers, mask)
    pruner.apply_mask()
    info = pruner.apply_rebuild()
    print(info)

    model(x)


def test_simplenet():
    net = SimpleNet()
    x = np.ones((1, 1, 9, 9), dtype=np.float32)

    net(x)

    assert net.conv1.W.shape == (3, 1, 3, 3)  # (oc, ic, kh, kw)
    assert net.bn1_1.gamma.shape == (3,)
    assert net.bn1_1.beta.shape == (3,)
    assert net.bn1_2.gamma.shape == (3,)
    assert net.bn1_2.beta.shape == (3,)
    assert net.bn1_3.gamma.shape == (3,)
    assert net.bn1_3.beta.shape == (3,)
    assert net.conv2.W.shape == (4, 3, 3, 3)  # (oc, ic, kh, kw)
    assert net.bn2.gamma.shape == (4,)
    assert net.bn2.beta.shape == (4,)
    assert net.fc.W.shape == (10, 4)  # (o, i)

    # mask
    target_channel = 1
    net.conv1.W.array[target_channel, ...] = 0

    graph = Graph(net, x)

    target_layers = ['/conv1']

    rebuild(net, graph, target_layers)

    assert net.conv1.W.shape == (2, 1, 3, 3)  # (oc, ic, kh, kw)
    assert net.bn1_1.gamma.shape == (2,)
    assert net.bn1_1.beta.shape == (2,)
    assert net.bn1_2.gamma.shape == (2,)
    assert net.bn1_2.beta.shape == (2,)
    assert net.bn1_3.gamma.shape == (2,)
    assert net.bn1_3.beta.shape == (2,)
    assert net.conv2.W.shape == (4, 2, 3, 3)  # (oc, ic, kh, kw)
    assert net.bn2.gamma.shape == (4,)
    assert net.bn2.beta.shape == (4,)
    assert net.fc.W.shape == (10, 4)

def test_no_target_layers():
    x = np.zeros((1, 3, 32, 32), dtype=np.float32)
    model = AllSupportedLayersNet()

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
