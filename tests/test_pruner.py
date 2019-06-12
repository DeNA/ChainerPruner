import numpy as np
import chainer

from chainerpruner import Graph
from chainerpruner.pruner import Pruner
from chainerpruner.masks.normmask import NormMask

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


def test_pruner():
    model = SimpleNet()
    x = np.ones((1, 1, 9, 9), dtype=np.float32)

    target_layers = ['conv1', 'conv2']
    percent = {'conv1': 0.6}
    default = 0.5

    graph = Graph(model, x)

    mask = NormMask(model, graph, target_layers, percent=percent, default=default)

    pruner = Pruner(model, x, target_layers, mask)

    print(model.count_params())

    pruner.mask()

    info = pruner.apply_rebuild()
    pruner.reinitialize()

    print(info)

    print(model.count_params())

    model(x)


def test_allnet():
    """
    - mask, rebuild, reinitialize
    """
    x = np.zeros((1, 1, 32, 32), dtype=np.float32)
    model = AllSupportedLayersNet()

    model(x)

    percent = 0.8
    target_layers = AllSupportedLayersNet.target_layers

    graph = Graph(model, x)
    mask = NormMask(model, graph, target_layers, percent=percent)
    pruner = Pruner(model, x, target_layers, mask)
    pruner.apply_mask()
    info = pruner.apply_rebuild()
    pruner.reinitialize()

    model(x)


def test_seq_out():

    class A(chainer.Chain):
        """N-output"""

        target_layers = [
            '/conv1',
            '/conv2',
        ]

        def __init__(self, n_class=10):
            super(A, self).__init__()
            with self.init_scope():
                self.conv1 = chainer.links.Convolution2D(None, 10, 3)
                self.bn1 = chainer.links.BatchNormalization(10)
                self.conv2 = chainer.links.Convolution2D(10, 3)
                self.bn2 = chainer.links.BatchNormalization(10)
                self.fc1 = chainer.links.Linear(None, n_class)
                self.fc2 = chainer.links.Linear(None, n_class)

        def __call__(self, x):
            h = self.conv1(x)
            h = self.bn1(h)
            h = self.conv2(h)
            h = self.bn2(h)
            h1 = self.fc1(h)
            h2 = self.fc2(h)
            return h1, h2

    x = np.zeros((1, 1, 32, 32), dtype=np.float32)
    model = A()

    model(x)

    percent = 0.8
    target_layers = A.target_layers

    graph = Graph(model, x)
    mask = NormMask(model, graph, target_layers, percent=percent)
    pruner = Pruner(model, x, target_layers, mask)
    pruner.apply_mask()
    info = pruner.apply_rebuild()
    pruner.reinitialize()

    model(x)


def test_map_out():

    class A(chainer.Chain):
        """N-output"""

        target_layers = [
            '/conv1',
            '/conv2',
        ]

        def __init__(self, n_class=10):
            super(A, self).__init__()
            with self.init_scope():
                self.conv1 = chainer.links.Convolution2D(None, 10, 3)
                self.bn1 = chainer.links.BatchNormalization(10)
                self.conv2 = chainer.links.Convolution2D(10, 3)
                self.bn2 = chainer.links.BatchNormalization(10)
                self.fc1 = chainer.links.Linear(None, n_class)
                self.fc2 = chainer.links.Linear(None, n_class)

        def __call__(self, x):
            h = self.conv1(x)
            h = self.bn1(h)
            h = self.conv2(h)
            h = self.bn2(h)
            h1 = self.fc1(h)
            h2 = self.fc2(h)
            return {'alice': h1, 'bob': h2}

    x = np.zeros((1, 1, 32, 32), dtype=np.float32)
    model = A()

    model(x)

    percent = 0.8
    target_layers = A.target_layers

    graph = Graph(model, x)
    mask = NormMask(model, graph, target_layers, percent=percent)
    pruner = Pruner(model, x, target_layers, mask)
    pruner.apply_mask()
    info = pruner.apply_rebuild()
    pruner.reinitialize()

    model(x)
