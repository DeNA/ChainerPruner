# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import links as L

from chainerpruner.rebuild.links.rebuildlink import RebuildLink
from chainerpruner.rebuild.links.utils import log_shape


class RebuildConvolution2D(RebuildLink):

    def active_rebuild(self, conv: L.Convolution2D):
        # mask_model is not None => conv-bnのbnを用いてconvをpruningする例外
        # ocに関する全kernelの総和が0のチャネルは除外し、それ以外を残す
        mask = conv.W.array.sum(axis=(1, 2, 3)) != 0
        self.logger.debug(log_shape(conv.W.array, mask))
        conv.W.array = conv.W.array[mask].copy()

        if conv.b is not None:
            self.logger.debug(log_shape(conv.b.array, mask))
            conv.b.array = conv.b.array[mask].copy()

        out_channels = conv.W.shape[0]
        conv.out_channels = out_channels

        return mask

    def passive_rebuild(self, conv, mask):
        # 出力チャネルに影響は無いので、biasは変化なし
        conv.W.array = conv.W.array[:, mask, :, :].copy()  # oc, ic, kh, kwのうちicをrebuild
        self.logger.debug(log_shape(conv.W.array, mask))

    def reinitialize(self, link: L.Convolution2D):
        out_channels, in_channels = link.W.shape[:2]
        link.out_channels = out_channels
        link._initialize_params(in_channels)
