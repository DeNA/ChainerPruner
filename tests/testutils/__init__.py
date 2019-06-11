import chainer
from chainer import functions as F
from chainercv.links.model.pickable_sequential_chain import PickableSequentialChain


class SimpleNet(chainer.Chain):

    def __init__(self, n_class=10):
        super(SimpleNet, self).__init__()
        with self.init_scope():
            self.conv1 = chainer.links.Convolution2D(None, 10, 3)
            self.bn1_1 = chainer.links.BatchNormalization(10)
            self.bn1_2 = chainer.links.BatchNormalization(10)
            self.bn1_3 = chainer.links.BatchNormalization(10)
            self.conv2 = chainer.links.Convolution2D(10, 3)
            self.bn2 = chainer.links.BatchNormalization(10)
            self.fc = chainer.links.Linear(None, n_class)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn1_1(h)
        h = self.bn1_2(h)
        h = self.bn1_3(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.fc(h)
        return h


class AllSupportedLayersNet(PickableSequentialChain):
    target_layers = [
        '/conv1',
        '/conv2',
    ]

    def __init__(self, n_class=10):
        super(AllSupportedLayersNet, self).__init__()
        with self.init_scope():
            self.conv1 = chainer.links.Convolution2D(None, 10, 3)
            self.bn1 = chainer.links.BatchNormalization(10)
            self.conv2 = chainer.links.Convolution2D(10, 3)
            self.bn2_1 = chainer.links.BatchNormalization(10)
            self.bn2_2 = chainer.links.BatchNormalization(10)
            self.conv3 = chainer.links.DepthwiseConvolution2D(10, 1, 3)
            self.fc1 = chainer.links.Linear(None, n_class)
            self.fc2 = chainer.links.Linear(None, n_class)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h_conv2 = self.conv2(h)
        h1 = self.bn2_1(h_conv2)
        h2 = self.bn2_2(h_conv2)
        h = h1 + h2
        h = self.conv3(h)
        h1 = self.fc1(h)
        h = F.average_pooling_2d(h, ksize=h.shape[2:])  # GAP
        h2 = self.fc2(h)
        h = h1 + h2
        return h
