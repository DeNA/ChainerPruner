import logging
import pytest

import numpy as np
import chainer

from chainerpruner.serializers import load_npz


class SimpleNet(chainer.Chain):
    
    def __init__(self, n):
        super(SimpleNet, self).__init__()
        with self.init_scope():
            self.conv1 = chainer.links.Convolution2D(1, n[0], 3)
            self.conv2 = chainer.links.Convolution2D(n[1], 3)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return h


def test_deserialize_cpu(tmpdir):
    weight_output = str(tmpdir.join('net.npz'))

    x = np.zeros((1, 1, 9, 9), dtype=np.float32)
    net_dst = SimpleNet([3, 2])
    net_src = SimpleNet([2, 3])

    net_dst(x)
    net_src(x)

    chainer.serializers.save_npz(weight_output, net_dst)
    load_npz(weight_output, net_src)
