# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.nn as nn

from chainerpruner.rebuild.rebuildlink import RebuildLink
from chainerpruner.rebuild.utils import log_shape


class RebuildLinear(RebuildLink):

    def passive_rebuild(self, linear, mask):
        # conv-linearだと影響を受ける

        self.logger.debug(log_shape(linear.weight.data, mask))

        input_shape = self.node.input_shape[0]
        if len(input_shape) == 4 and input_shape[1] == len(mask):
            # prev node is conv: conv-fc
            n_out, n_in = linear.weight.shape
            w = linear.weight.data.clone().reshape(n_out, *input_shape[1:])
            w = w[:, mask, :, :]
            w = w.reshape(n_out, -1)
        else:
            # conv-gap-fc, conv-view(flatten)-fc
            n_out, n_in = linear.weight.shape
            pixels_per_channel = input_shape[1] // len(mask)

            assert mask.dim() == 1

            # TODO(tkat0) refactor
            # convert channel mask to pixel-level mask
            flatten_mask = torch.zeros((n_in,), dtype=torch.uint8, device=mask.device, requires_grad=False)
            for i, m in enumerate(mask):
                m = int(m)
                if m == 1:
                    begin, end = i * pixels_per_channel, (i + 1) * pixels_per_channel
                    flatten_mask[begin:end] = m

            w = linear.weight.data[:, flatten_mask].clone()

        linear.weight.data = w

    def reinitialize(self, link: nn.Linear):
        # _, in_size = link.W.shape
        # link._initialize_params(in_size)
        raise NotImplementedError()

    def update_attributes(self, link: nn.Linear):
        out_features, in_features = link.weight.shape

        link.in_features = in_features
        link.out_features = out_features
