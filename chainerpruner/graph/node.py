# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]


class Node():
    """計算グラフのノード。channel pruningに必要な情報を抽出している。

    入出力のテンソルのサイズや、ノードの型、chainerのLinkやFuntionのオブジェクト自体もアクセスできる
    """

    def __init__(self, id_, type_, args=None, out=None, link=None, function=None, module=None):
        """

        nextは、List[Node or List[Node]]

        Args:
            id_:
            type_:
            args:
            out:
            link:
            function:
        """

        if not args:
            args = []
        if not out:
            out = []

        self.name = id_
        self.id = id_
        self.type = type_
        self.input_id = [id(a) for a in args]
        self.input_shape = [a.shape if hasattr(a, 'shape') else None for a in args]
        self.output_id = [id(o) for o in out if o is not None]
        self.output_shape = [o.shape if hasattr(o, 'shape') else None for o in out]
        self.link = link
        self.function = function
        # self.module = module

    def __repr__(self):
        if self.link:
            node_class = 'link'
        elif self.function:
            node_class = 'function'
        else:
            node_class = 'unknown'
        return 'Node(name=\'{}\', id={}, type={}, input={},' \
               ' output={}, input_shape={}, output_shape={}, class={})'.format(
            self.name,
            self.id,
            self.type.__name__,
            self.input_id,
            self.output_id,
            self.input_shape,
            self.output_shape,
            node_class)
