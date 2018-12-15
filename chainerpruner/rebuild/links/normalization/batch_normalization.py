# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import links as L

from chainerpruner.rebuild.links.rebuildlink import RebuildLink
from chainerpruner.rebuild.links.utils import log_shape


class RebuildBatchNormalization(RebuildLink):

    def _rebuild(self, bn, mask):
        self.logger.debug(log_shape(bn.gamma.array, mask))
        bn.gamma.array = bn.gamma.array[mask].copy()

        w = bn.beta
        if w is not None:
            bn.beta.array = w.array[mask].copy()

        bn.avg_var = bn.avg_var[mask].copy()
        bn.avg_mean = bn.avg_mean[mask].copy()

        return mask

    def active_rebuild(self, bn: L.BatchNormalization):
        w = bn.gamma
        mask = w.array != 0
        self._rebuild(bn, mask)
        return mask

    def passive_rebuild(self, bn, mask):
        self._rebuild(bn, mask)

    def reinitialize(self, link: L.BatchNormalization):
        link._initialize_params(link.gamma.shape)
