# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from typing import Sequence
from collections import OrderedDict

import chainer
from chainerpruner.trace import TraceLinkHook, TraceFunctionHook
from chainerpruner.node import Node


class Graph():
    """Computation Graph Parser
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
        with chainer.using_config('train', False), chainer.force_backprop_mode():
            with TraceLinkHook() as link_hook:
                out = model(*args)

            with TraceFunctionHook() as func_hook:
                out.grad = xp.ones_like(out.array)
                out.backward()

        self.links = link_hook.graph  # type: Sequence[Node]
        # get global link name
        mapping = {id(link): name for name, link in model.namedlinks()}
        def replace_name(node):
            node.name = mapping[node.id]
            return node
        self.links = [replace_name(node) for node in self.links]
        self.links = OrderedDict([(node.name, node) for node in self.links])

        self.functions = func_hook.graph  # type: Sequence[Node]
        self.functions = reversed(self.functions)
        self.functions = OrderedDict([(node.name, node) for node in self.functions])

        self._prepare()

    def _prepare(self):
        # TODO(tkato) 高速化
        # linkとfunctionの解析結果をLinkメインでマージする
        # pruningのため、linkの各ノードの後続のlinkをセットする
        # linkの次のlinkを探索してセットする

        def get_next_functions(output_id, next_function_output_ids):
            """後続のFunctionを再帰的にたどる"""
            for node in self.functions.values():
                if set(output_id) & set(node.input_id):
                    next_function_output_ids.append(node.output_id)
                    get_next_functions(node.output_id, next_function_output_ids)
                    return

        for node in self.links.values():
            next_links = []

            # 直接nodeの出力がつながるlinkを探す
            # linkのout, inのidが一致するものをnext linkとする
            for next_node in self.links.values():
                if node.output_id == next_node.input_id:
                    next_links.append(next_node.name)

            # 後続のfunctionをたどる
            next_function_output_ids = []
            get_next_functions(node.output_id, next_function_output_ids)

            # nodeに後続するfunctionsのlinkにおける名前を取得する
            for next_function_id in next_function_output_ids:
                for next_node in self.links.values():
                    if set(next_function_id) & set(next_node.input_id):
                        next_links.append(next_node.name)

            node.next = next_links
