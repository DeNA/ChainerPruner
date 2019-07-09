# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from collections import deque
from typing import Mapping
from collections import OrderedDict

from chainerpruner.graph import Graph


def calc_pruning_connection(graph: Graph) -> Mapping:
    """calculate link connections for channel pruning

    Args:
        model:
        args:

    Returns:

    """

    if graph.is_chainer:
        from chainerpruner.rebuild.chainer.same_io_channels_layers import is_connected_io_channels
    else:
        from chainerpruner.rebuild.pytorch.same_io_channels_layers import is_connected_io_channels

    pruning_connection_info = OrderedDict()

    # 現在のノードのpruningで影響がある後続するlinkノードをすべて取得する
    # TODO(tkat0) linkとfunctionをマージしているため、L.Convolution2DとConvolution2DFunctionのパスが独立し多重辺ができる
    #   そのため、後続のノードの依存をチェックするis_connected_io_channelsがlinkとfunctionそれぞれ対応する必要があり複雑化
    #   FunctionNodeベースでグラフを作りつつ、pruningのためにweightの情報を付与する方がシンプルになりそう
    for node in graph.graph.nodes:  # all node
        # TODO(tkat0) ここにパスごとのlistをいれるようにすると管理しやすいかもしれない
        #   今は全パスがflatten
        affected_nodes = []

        # 同一nodeは1度だけaffected_nodesに追加するようにする。
        # 現在の実装では多重辺が存在するので重複する場合がある
        visited = set()

        if node.link is None:
            continue  # function

        stack = deque([])
        stack.extend(graph.graph.successors(node))
        while len(stack) > 0:
            next_node = stack.pop()
            if next_node.link and next_node.name not in visited:
                affected_nodes.append(next_node.name)
                visited.add(next_node.name)
            if not is_connected_io_channels(next_node):
                continue  # to next path
            stack.extend(graph.graph.successors(next_node))

        if affected_nodes:
            affected_nodes.reverse()
            pruning_connection_info[node.name] = affected_nodes

    # この時点ではResBlockやConv2DBNActivのようなUserDefinedLinkも含まれる

    # for example
    # conv1-bn1-conv2-fc
    # {'conv1': ['bn1', 'conv2'], 'conv2': ['bn2', 'fc']}
    return pruning_connection_info
