# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from typing import Mapping
from collections import OrderedDict

from chainerpruner.rebuild.same_io_channels_layers import same_io_channels_layers
from chainerpruner.graph import Graph


def calc_pruning_connection(graph: Graph) -> Mapping:
    """calculate link connections for channel pruning

    Args:
        model:
        args:

    Returns:

    """

    pruning_connection_info = OrderedDict()

    for node in graph.links.values():
        affected_nodes = []
        for next_name in node.next:
            next_node = graph.links[next_name]
            affected_nodes.append(next_name)
            if next_node.type not in same_io_channels_layers:
                break
        if affected_nodes:
            pruning_connection_info[node.name] = affected_nodes

    # TODO(tkato) skip-connectionやbnなど、pruning対象外のkeyを削除する

    # for example
    # conv1-bn1-conv2-fc
    # {'conv1': ['bn1', 'conv2'], 'conv2': ['bn2', 'fc']}
    return pruning_connection_info
