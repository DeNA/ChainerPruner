# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from typing import Sequence
import weakref
import chainer

from chainerpruner.node import Node


class TraceFunctionHook(chainer.FunctionHook):
    name = 'TraceFunctionHook'

    def __init__(self):
        self.graph = []

    def backward_postprocess(self, function, in_data, out_grad):
        args = function.inputs
        out = function.outputs

        # get entity for backward
        out = [o if not isinstance(o, weakref.ref) else o() for o in out]

        node = Node(id_=id(function),
                    type_=type(function),
                    args=args,
                    out=out,
                    function=function)

        self.graph.append(node)


class TraceLinkHook(chainer.LinkHook):
    name = 'TraceLinkHook'

    def __init__(self):
        self.graph = []

    def forward_postprocess(self, args):
        link = args.link
        out = args.out
        args = args.args

        if not isinstance(out, Sequence):
            out = [out]

        args = [a.node for a in args]
        out = [o.node for o in out]

        node = Node(id_=id(link),
                    type_=type(link),
                    args=args,
                    out=out,
                    link=link)

        self.graph.append(node)
