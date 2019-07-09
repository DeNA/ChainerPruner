# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import links as L

from chainerpruner.rebuild.rebuildlink import RebuildLink
from chainerpruner.rebuild.utils import log_shape


class RebuildConvolution2D(RebuildLink):

    def update_attributes(self, conv: L.Convolution2D):
        conv.out_channels = conv.W.shape[0]

    def active_rebuild(self, conv: L.Convolution2D):
        mask = conv.W.array.sum(axis=(1, 2, 3)) != 0
        self.logger.debug(log_shape(conv.W.array, mask))
        conv.W.array = conv.W.array[mask].copy()

        if conv.b is not None:
            self.logger.debug(log_shape(conv.b.array, mask))
            conv.b.array = conv.b.array[mask].copy()

        return mask

    def passive_rebuild(self, conv, mask):
        # 出力チャネルに影響は無いので、biasは変化なし
        conv.W.array = conv.W.array[:, mask, :, :].copy()  # oc, ic, kh, kwのうちicをrebuild
        self.logger.debug(log_shape(conv.W.array, mask))

    def reinitialize(self, link: L.Convolution2D):
        out_channels, in_channels = link.W.shape[:2]
        link.out_channels = out_channels
        link._initialize_params(in_channels)
