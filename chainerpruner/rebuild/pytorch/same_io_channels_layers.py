# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import torch.nn as nn

# TODO(tkat0) aggregate functions
_connected_io_channels_layers = [
    nn.BatchNorm2d,
    nn.ReLU6,
    nn.ReLU,
    nn.Upsample,
    nn.MaxPool2d,
]


def is_connected_io_channels(node):
    """入出力のチャネル数が連動している層の場合Trueを返す

    Args:
        node:

    Returns:

    """
    if node.type == nn.Conv2d and node.link.groups > 1:
        # Depthwise Convolution
        return True
    elif node.type in _connected_io_channels_layers:
        return True
    else:
        return False
