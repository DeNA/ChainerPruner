# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import torch.nn as nn


_disconnected_io_channels_layers = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Linear,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
]


def is_connected_io_channels(node):
    """Return True if the number of input and output channels of node is linked

    Args:
        node:

    Returns:

    """
    if node.type == nn.Conv2d and node.link.groups > 1:
        # Depthwise Convolution
        return True
    elif node.type in _disconnected_io_channels_layers:
        return False
    else:
        return True  # default
