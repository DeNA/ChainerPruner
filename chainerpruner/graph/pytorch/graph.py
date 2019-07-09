# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from typing import Sequence
import networkx as nx

from chainerpruner.graph.pytorch.trace import ModuleTraceHook
from chainerpruner.graph.node import Node


class PyTorchGraph():
    """Computation Graph Parser
    """

    def __init__(self, model, args):
        # convert to tuple of Variable
        if not isinstance(args, Sequence):
            args = [args]

        # 計算グラフを構築しながら、hookを利用して各Moduleの情報を取得する
        hook = ModuleTraceHook(model)
        model(*args)

        nodes = hook.graph  # type: Sequence[Node]
        self.graph = self._traverse_connections(nodes)

    def _traverse_connections(self, nodes):
        # Hookで得たlinkとfunctionのList[Node]をマージしてグラフにする

        # node.nameをキーにしたGraph
        # node attrでnodeの実体へアクセス
        graph = nx.DiGraph()

        for node in nodes:
            graph.add_node(node)
            for next_node in nodes:
                if set(node.output_id) & set(next_node.input_id):
                    graph.add_edge(node, next_node)

        return graph
