# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn

from chainerpruner.rebuild.rebuildlink import RebuildLink
from chainerpruner.rebuild.utils import log_shape


class RebuildConvolution2D(RebuildLink):

    def active_rebuild(self, conv: nn.Conv2d):
        assert conv.groups == 1, 'Group Convolution is not supported.'

        # mask_model is not None => conv-bnのbnを用いてconvをpruningする例外
        # ocに関する全kernelの総和が0のチャネルは除外し、それ以外を残す
        mask = conv.weight.data.sum(dim=(1, 2, 3)) != 0
        self.logger.debug(log_shape(conv.weight.data, mask))
        conv.weight.data = conv.weight.data[mask].clone()

        if conv.bias is not None:
            self.logger.debug(log_shape(conv.bias.data, mask))
            conv.bias.data = conv.bias.data[mask].clone()

        out_channels = int(conv.weight.shape[0])
        conv.out_channels = out_channels

        return mask

    def passive_rebuild(self, conv, mask):
        # 出力チャネルに影響は無いので、biasは変化なし
        if conv.groups > 1:
            # Depthwise Convolution
            conv.weight.data = conv.weight.data[mask, :, :, :].clone()
            self.logger.debug(log_shape(conv.weight.data, mask))
            in_channels = int(conv.weight.shape[0])
            conv.groups = in_channels
            out_channels = conv.groups * int(conv.weight.shape[1])
            conv.in_channels = in_channels
            conv.out_channels = out_channels
        else:
            conv.weight.data = conv.weight.data[:, mask, :, :].clone()  # oc, ic, kh, kwのうちicをrebuild
            self.logger.debug(log_shape(conv.weight.data, mask))
            in_channels = int(conv.weight.shape[1])
            conv.in_channels = in_channels

    def reinitialize(self, link: nn.Conv2d):
        # out_channels, in_channels = link.conv.weight.shape[:2]
        # link.out_channels = out_channels
        # link._initialize_params(in_channels)
        raise NotImplementedError()

    def update_attributes(self, conv: nn.Conv2d):
        if conv.groups > 1:
            # Depthwise Convolution
            in_channels = int(conv.weight.shape[0])
            conv.groups = in_channels
            out_channels = conv.groups * int(conv.weight.shape[1])
            conv.in_channels = in_channels
            conv.out_channels = out_channels
        else:
            in_channels = int(conv.weight.shape[1])
            conv.in_channels = in_channels
