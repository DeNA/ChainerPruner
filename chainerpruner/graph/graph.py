# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainerpruner import utils


class Graph():
    """Computation Graph Parser

    Chainerの計算グラフはLinkとFunctionから構成される
    Channel Pruningする上では、現在の層をPruningしたときに
    後続する層のどこまでその影響があるかを知ることが重要になる
    ここではモデルをパースして、linksに各層とその影響する層の情報をList[Node]として格納している
    """

    def __init__(self, model, args):
        if utils.is_chainer_model(model):
            from chainerpruner.graph.chainer.graph import ChainerGraph
            graph = ChainerGraph(model, args)
            self.is_chainer = True
            self.is_pytorch = False
        else:
            from chainerpruner.graph.pytorch.graph import PyTorchGraph
            graph = PyTorchGraph(model, args)
            self.is_chainer = False
            self.is_pytorch = True

        self.graph = graph.graph

    def plot(self, options=None):
        import networkx as nx
        if not options:
            options = dict()
        nx.draw(self.graph, with_label=True, **options)
