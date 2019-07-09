# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from typing import Sequence

from chainerpruner.graph.node import Node


class ModuleTraceHook():
    """modelをトレースしてpruningに必要な計算グラフの情報を取得する

    トレース後のグラフは`graph`属性として得られる

    """

    def __init__(self, model):
        self.graph = []
        # グラフ内各moduleをuniqueに特定するための名前
        self._module_global_names = dict()
        self._regist(model)

    def _regist(self, model):
        for name, module in model.named_modules():
            # nn.Sequentialやユーザー定義の複数のmoduleをもつmoduleではなく、
            # Conv2dなどの単一のmoduleのみ解析したいため
            n_modules = len(list(module.modules()))
            if n_modules == 1:
                # print('regist {}'.format(name))
                self._module_global_names[module] = name
                module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        """

        Args:
            module:
            input: tuple(Tensor)
            output: Tensor

        Returns:

        """
        if not isinstance(input, Sequence):
            input = [input]
        if not isinstance(output, Sequence):
            output = [output]

        name = self._module_global_names[module]

        node = Node(id_=name,
                    type_=type(module),
                    args=None,
                    out=None,
                    # module=module,
                    link=module,
                    )

        # TODO(tkat0) mv
        node.input_id = [i._cdata for i in input]
        node.input_shape = [tuple(i.shape) if hasattr(i, 'shape') else None for i in input]
        node.output_id = [o._cdata for o in output if o is not None]
        node.output_shape = [tuple(o.shape) if hasattr(o, 'shape') else None for o in output]

        if set(node.input_id) == set(node.output_id):
            pass  # skip in-place op (like ReLU)
        else:
            self.graph.append(node)
