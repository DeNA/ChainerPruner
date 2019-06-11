import numpy as np
import chainer

from chainerpruner.masks.normmask import NormMask
from chainerpruner import Graph


class SimpleNet(chainer.Chain):

    def __init__(self):
        super(SimpleNet, self).__init__()
        with self.init_scope():
            self.conv1 = chainer.links.Convolution2D(1, 10, 3)
            self.bn1_1 = chainer.links.BatchNormalization(10)
            self.bn1_2 = chainer.links.BatchNormalization(10)
            self.bn1_3 = chainer.links.BatchNormalization(10)
            self.conv2 = chainer.links.Convolution2D(10, 3)
            self.bn2 = chainer.links.BatchNormalization(10)
            self.fc = chainer.links.Linear(10)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn1_1(h)
        h = self.bn1_2(h)
        h = self.bn1_3(h)
        h = self.conv2(h)
        h = self.bn2(h)
        # h = chainer.functions.average_pooling_2d(h, ksize=h.shape[2:])  # GAP
        h = self.fc(h)
        return h


def test_normmask():

    model = SimpleNet()
    x = np.ones((1, 1, 9, 9), dtype=np.float32)

    target_layers = ['/conv1', '/conv2']
    percent = {'/conv1': 0.8}
    default = 0.4

    graph = Graph(model, x)

    mask = NormMask(model, graph, target_layers, percent=percent, default=default)

    info = mask()

    assert info[0]['name'] == '/conv1'
    assert info[0]['before'] == 10
    assert info[0]['after'] == 2
    assert info[1]['name'] == '/conv2'
    assert info[1]['before'] == 10
    assert info[1]['after'] == 6

