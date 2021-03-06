# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import links as L

from chainerpruner.rebuild.rebuildlink import RebuildLink
from chainerpruner.rebuild.utils import log_shape


class RebuildLinear(RebuildLink):

    def update_attributes(self, link: L.Linear):
        link.out_size, link.in_size = link.W.shape

    def passive_rebuild(self, linear, mask):

        self.logger.debug(log_shape(linear.W.array, mask))

        input_shape = self.node.input_shape[0]
        if len(input_shape) == 4 and input_shape[1] == len(mask):
            # conv-fc
            n_out, n_in = linear.W.shape
            w = linear.W.array.copy().reshape(n_out, *input_shape[1:])
            w = w[:, mask, :, :]
            w = w.reshape(n_out, -1)
        else:
            # conv-gap-fc
            w = linear.W.array[:, mask].copy()

        linear.W.array = w

    def reinitialize(self, link: L.Linear):
        _, in_size = link.W.shape
        link._initialize_params(in_size)
