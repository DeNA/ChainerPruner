# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from typing import Mapping
from collections import OrderedDict

from chainerpruner.rebuild.same_io_channels_layers import is_connected_io_channels
from chainerpruner.graph import Graph


def calc_pruning_connection(graph: Graph) -> Mapping:
    """calculate link connections for channel pruning

    Args:
        model:
        args:

    Returns:

    """

    pruning_connection_info = OrderedDict()

    # 現在のノードのpruningで影響がある後続するlinkノードをすべて取得する
    # TODO(tkat0) linkとfunctionをマージしているため、L.Convolution2DとConvolution2DFunctionのパスが独立し多重辺ができる
    #   そのため、後続のノードの依存をチェックするis_connected_io_channelsがlinkとfunctionそれぞれ対応する必要があり複雑化
    #   FunctionNodeベースでグラフを作りつつ、pruningのためにweightの情報を付与する方がシンプルになりそう
    for node in graph.graph.nodes:  # all node
        affected_nodes = []

        # 同一nodeは1度だけaffected_nodesに追加するようにする。
        # 現在の実装では多重辺が存在するので重複する場合がある
        visited = set()

        if node.link is None:
            continue  # function

        def get_affected_nodes(node):
            """再帰的に深さ方向へ影響有るノードを探索"""
            for next_node in graph.graph.successors(node):
                if next_node.link and next_node.name not in visited:
                    affected_nodes.append(next_node.name)
                    visited.add(next_node.name)
                if not is_connected_io_channels(next_node):
                    continue  # to next path
                # TODO(tkat0) ここにパスごとのlistをいれるようにすると管理しやすいかもしれない
                #   今は全パスがflatten
                get_affected_nodes(next_node)

        get_affected_nodes(node)

        if affected_nodes:
            pruning_connection_info[node.name] = affected_nodes

    # この時点ではResBlockやConv2DBNActivのようなUserDefinedLinkも含まれる
    print(pruning_connection_info)

    # for example
    # conv1-bn1-conv2-fc
    # {'conv1': ['bn1', 'conv2'], 'conv2': ['bn2', 'fc']}
    return pruning_connection_info
