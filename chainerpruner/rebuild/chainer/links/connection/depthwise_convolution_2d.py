# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import links as L

from chainerpruner.rebuild.rebuildlink import RebuildLink
from chainerpruner.rebuild.utils import log_shape


class RebuildDepthwiseConvolution2D(RebuildLink):

    def update_attributes(self, conv):
        # do nothing
        return

    def active_rebuild(self, conv):
        # 通常、depthwise-convはpriningしない。sep-conv(convdw-convpw)の場合、convpwをpruningする
        raise NotImplementedError

    def passive_rebuild(self, conv: L.DepthwiseConvolution2D, mask):
        self.logger.debug(log_shape(conv.W.array, mask))
        conv.W.array = conv.W.array[:, mask].copy()
        if conv.b is not None:
            conv.b.array = conv.b.array[mask].copy()

    def reinitialize(self, link: L.DepthwiseConvolution2D):
        _, in_channels = link.W.shape[:2]
        link._initialize_params(in_channels)

    def custom_calculator(self, in_data, **kwargs):
        # TODO(tkato)
        raise NotImplementedError()
