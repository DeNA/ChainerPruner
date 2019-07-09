# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from typing import Sequence, Mapping
import networkx as nx

import chainer
from chainerpruner.graph.chainer.trace import TraceLinkHook, TraceFunctionHook
from chainerpruner.graph.node import Node


class ChainerGraph():
    """Computation Graph Parser

    Chainerの計算グラフはLinkとFunctionから構成される
    Channel Pruningする上では、現在の層をPruningしたときに
    後続する層のどこまでその影響があるかを知ることが重要になる
    ここではモデルをパースして、linksに各層とその影響する層の情報をList[Node]として格納している
    """

    def __init__(self, model, args):
        xp = model.xp

        # convert to tuple of Variable
        if not isinstance(args, Sequence):
            args = [args]
        args = [chainer.Variable(a) if not isinstance(a, chainer.Variable) else a for a in args]

        # linkとfunctionそれぞれの計算グラフ情報を取得する
        # pruningはLinkに着目したほうが実装がシンプルになるが、conv-bn-gap-fcのようなFunctionを挟むケースも対応するため、
        # それぞれ解析し、結果をVariableNodeのidを用いてマージしている
        # ここはもっとエレガントになるように変更予定
        with chainer.using_config('train', False), chainer.force_backprop_mode():
            with TraceLinkHook() as link_hook:
                outs = model(*args)
            if isinstance(outs, Mapping):
                outs = list(outs.values())
            if not isinstance(outs, Sequence):
                outs = [outs]

            with TraceFunctionHook() as func_hook:
                for out in outs:
                    out.grad = xp.ones_like(out.array)
                    out.backward()

        self.links = link_hook.graph  # type: Sequence[Node]
        # get global link name
        mapping = {id(link): name for name, link in model.namedlinks()}
        def replace_name(node):
            node.name = mapping[node.id]
            return node
        self.links = [replace_name(node) for node in self.links]

        self.functions = func_hook.graph  # type: Sequence[Node]
        self.functions = list(reversed(self.functions))

        nodes = list()
        nodes.extend(self.links)
        nodes.extend(self.functions)

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
