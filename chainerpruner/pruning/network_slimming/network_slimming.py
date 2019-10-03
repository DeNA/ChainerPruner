# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainerpruner import Graph
from chainerpruner.rebuild import rebuild
from chainerpruner.masks import NormMask


def pruning(model, args, target_conv_layers, threshold, default=None):
    """Apply mask and rebuild for Network Slimming

    Args:
        model (torch.nn.Module, chainer.Chain): target model.
        args: dummy inputs of target model.
        target_conv_layers (list[str]):
        threshold (float, dict): mask threshold for BatchNorm2d.weight.
        default (float, Optional): default threshold (available only if threshold is dict).

    Returns:
        dict: pruning runtime information

    """

    graph = Graph(model, args)
    mask = NormMask(model, graph, target_conv_layers, threshold=threshold, default=default,
                    mask_layer='batchnorm')
    info = {}
    info['mask'] = mask()
    info['rebuild'] = rebuild(model, graph, target_conv_layers)
    return info
