# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from collections import Sequence

import chainer


class Node():
    def __init__(self, id_, type_, args, out, name=None):
        """

        Args:
            name:
            id_:
            type_:
            args:
            out:
        """

        if not all([isinstance(a, chainer.variable.VariableNode) for a in args]):
            raise TypeError()
        if not all([isinstance(o, chainer.variable.VariableNode) for o in out]):
            raise TypeError()

        self.name = id_
        self.id = id_
        self.type = type_
        self.input_id = [id(a) for a in args]
        self.input_shape = [a.shape if hasattr(a, 'shape') else None for a in args]
        self.output_id = [id(o) for o in out if o is not None]
        self.output_shape = [o.shape if hasattr(o, 'shape') else None for o in out]
        self.next = []

    def __repr__(self):
        return 'Node([{}] name=\'{}\', type={}, input={},' \
               ' output={}, input_shape={}, output_shape={}, next={})'.format(self.id, self.name,
                                                                     self.type.__name__,
                                                                     self.input_id,
                                                                     self.output_id,
                                                                     self.input_shape,
                                                                     self.output_shape,
                                                                     self.next)
