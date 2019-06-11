from collections import OrderedDict
import numpy as np
import chainer
from chainer import functions as F

from chainerpruner.rebuild.calc_pruning_connection import calc_pruning_connection
from chainerpruner.graph import Graph


class SimpleNet(chainer.Chain):

    def __init__(self):
        super(SimpleNet, self).__init__()
        with self.init_scope():
            self.conv1 = chainer.links.Convolution2D(1, 3, 3)
            self.bn1_1 = chainer.links.BatchNormalization(3)
            self.bn1_2 = chainer.links.BatchNormalization(3)
            self.bn1_3 = chainer.links.BatchNormalization(3)
            self.conv2 = chainer.links.Convolution2D(2, 3)
            self.bn2 = chainer.links.BatchNormalization(2)
            self.fc = chainer.links.Linear(10)

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


def test_calc_pruning_connection():
    x = np.zeros((1, 1, 9, 9), dtype=np.float32)
    net = SimpleNet()

    graph = Graph(net, x)

    pruning_connection_info = calc_pruning_connection(graph)

    # TODO(tkato) bnのkeyは消しても良いかもしれない 仕様として
    t = OrderedDict([
        ('/conv1', ['/bn1_1', '/bn1_2', '/bn1_3', '/conv2']),
        ('/bn1_1', ['/bn1_2', '/bn1_3', '/conv2']),
        ('/bn1_2', ['/bn1_3', '/conv2']),
        ('/bn1_3', ['/conv2']),
        ('/conv2', ['/bn2', '/fc']),
        ('/bn2', ['/fc'])
    ])

    assert pruning_connection_info == t
