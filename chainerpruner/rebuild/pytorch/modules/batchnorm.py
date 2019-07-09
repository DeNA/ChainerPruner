# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import torch.nn as nn

from chainerpruner.rebuild.rebuildlink import RebuildLink
from chainerpruner.rebuild.utils import log_shape


class RebuildBatchNormalization(RebuildLink):

    def _rebuild(self, bn, mask):
        self.logger.debug(log_shape(bn.weight.data, mask))
        bn.weight.data = bn.weight.data[mask].clone()

        w = bn.bias
        if w is not None:
            bn.bias.data = w.data[mask].clone()

        bn.running_mean = bn.running_mean[mask].clone()
        bn.running_var = bn.running_var[mask].clone()

        bn.num_features = len(bn.weight)

        return mask

    def active_rebuild(self, bn: nn.BatchNorm2d):
        w = bn.weight
        mask = w.data != 0
        self._rebuild(bn, mask)
        return mask

    def passive_rebuild(self, bn, mask):
        self._rebuild(bn, mask)

    def reinitialize(self, link: nn.BatchNorm2d):
        # link._initialize_params(link.weight.shape)
        raise NotImplementedError()

    def update_attributes(self, bn: nn.BatchNorm2d):
        bn.num_features = len(bn.weight)
