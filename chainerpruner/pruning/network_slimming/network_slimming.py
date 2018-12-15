# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainerpruner import Graph
from chainerpruner.rebuild import rebuild
from chainerpruner.rebuild import calc_pruning_connection
from chainerpruner.masks import NormMask


def pruning(model, args, target_layers, threshold, default=None):

    graph = Graph(model, args)
    mask = NormMask(model, graph, target_layers, threshold=threshold, default=default,
                    mask_layer='batchnorm')
    info = {}
    info['mask'] = mask()
    info['rebuild'] = rebuild(model, graph, target_layers)
    return info
