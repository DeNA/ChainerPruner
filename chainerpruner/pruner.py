# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from typing import Sequence, Union, Tuple

from chainerpruner.mask import Mask
from chainerpruner.rebuild.rebuild import rebuild
from chainerpruner.rebuild import reinitialize_model
from chainerpruner.rebuild.calc_pruning_connection import calc_pruning_connection
from chainerpruner.graph import Graph
from chainerpruner.testing.typing import NdArray


class Pruner():
    """High-level Pruning API
    """

    def __init__(self, model, args: Union[NdArray, Tuple[NdArray]],
                 target_layers: Sequence, mask: Mask, mapping=None):
        self.model = model
        self.target_layers = target_layers
        self.mask = mask
        self._mapping = mapping

        self.graph = Graph(model, args)
        self._pruning_connection_info = calc_pruning_connection(self.graph)

    def apply_mask(self):
        return self.mask()

    def apply_rebuild(self):
        return rebuild(self.model, self.graph, self.target_layers, mapping=self._mapping)

    def reinitialize(self):
        reinitialize_model(self.model, self._mapping)

    def update_mask_fun(self, mask: Mask):
        self.mask = mask
