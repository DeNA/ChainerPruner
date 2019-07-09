# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import links as L
from chainer.functions.connection.convolution_2d import Convolution2DFunction
from chainer.functions.connection.linear import LinearFunction

# TODO(tkat0) これらの実装はRebuildLink側に統合しても綺麗かもしれない

_connected_io_channels_layers = [
    L.DepthwiseConvolution2D,
    L.BatchNormalization,
]

_disconnected_io_channels_layers = [
    Convolution2DFunction,
    LinearFunction,
]

def is_grouped_convolution(node):
    if node.type == L.Convolution2D and node.link.groups > 1:
        return True
    elif node.type == Convolution2DFunction and node.function.groups > 1:
        return True
    else:
        return False

def is_connected_io_channels(node):
    """入出力のチャネル数が連動している層の場合Trueを返す

    Args:
        node:

    Returns:

    """
    # TODO(tkat0) ここもカスタマイズ可能に
    if is_grouped_convolution(node):
        return True
    elif node.type in _connected_io_channels_layers:
        return True
    elif node.type in _disconnected_io_channels_layers:
        # 主にFunctionは異なるin/outの層は少ないのでblacklist式にしている
        return False
    elif node.function:
        # 全てのFunctionはin/outチャネルが一致ていると仮定
        return True
    else:
        return False
