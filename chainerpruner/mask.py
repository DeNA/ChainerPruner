# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import logging
import numpy as np

from chainer import links as L

from chainerpruner.rebuild import calc_pruning_connection

logger = logging.getLogger(__name__)


class Mask():
    """
    """

    def __init__(self, model, graph, target_layers, mask_layer=None):
        self.model = model
        self._model_dict = {name: link for name, link in model.namedlinks()}
        self.graph = graph
        self.target_layers = target_layers
        self.logger = logger
        self.pruning_connection_info = calc_pruning_connection(graph)
        self.masks = dict()
        self._mask_layer = mask_layer

        self._hooks = []

        cand_mask_layer = [None, 'batchnorm']
        if mask_layer not in cand_mask_layer:
            raise AttributeError('mask_layer is expected which {}'.format(cand_mask_layer))

    def get_filter_norm(self, mask):
        """get mask for pruning

        Args:
            mask:

        Returns:

        """
        raise NotImplementedError()

    def add_hook(self, hook_fn):
        self._hooks.append(hook_fn)

    def get_thresholds(self, name, mask):
        """

        Args:
            name: pruning target layer
            mask:

        Returns:
            pruning threshold
        """
        raise NotImplementedError()

    def _get_mask(self, name):
        conv = self._model_dict[name]
        if self._mask_layer is None:
            mask = conv.W.array.copy()
        elif self._mask_layer == 'batchnorm':
            # conv-bn
            post_conv_bn_name = self.pruning_connection_info[name][0]
            bn = self._model_dict[post_conv_bn_name]
            if not isinstance(bn, L.BatchNormalization):
                raise ValueError('expected {}(Conv) -> {}(BatchNorm)'.format(name, post_conv_bn_name))
            mask = bn.gamma.array.copy()
            mask = mask.reshape(-1, 1, 1, 1) # to mask conv weight (oc, ic, kh, kw)
        return conv.W.array, mask

    def __call__(self):
        info = []

        mask_count = 0

        self.logger.debug('target_layers: %s', self.target_layers)

        # get mask vector
        target_weights = []
        for name, link in self.model.namedlinks(skipself=True):

            self.logger.debug('name: %s', name)

            if name not in self.target_layers:
                continue

            target_weight, mask = self._get_mask(name)
            target_weights.append(target_weight)

            out_channels = mask.shape[0]
            mask = self.get_filter_norm(mask)
            if mask.shape != (out_channels, 1, 1, 1):
                raise RuntimeError()

            self.masks[name] = mask

        for hook in self._hooks:
            hook()

        # apply mask
        for target_weight, (name, mask_) in zip(target_weights, self.masks.items()):

            threshold = self.get_thresholds(name, mask_)

            self.logger.debug('mask: %s, threshold: %s', name, threshold)

            # apply mask
            mask = mask_ >= threshold  # 0: pruning, 1: non-pruning
            mask = mask.astype(np.float32)

            info_ = {
                'name': name,
                'before': len(mask),
                'after': int(sum(mask)),
                'threshold': threshold,
            }
            info.append(info_)

            if int(sum(mask)) == 0:
                raise ValueError('channels after masked is zero. please fix threshold [{}, {}].\ninfo: {}'.format(
                    float(mask_.min()), float(mask_.max()),
                    info_
                ))

            # to zero
            target_weight *= mask

            mask_count += 1

        if mask_count == 0:
            logger.warning('mask count == 0')

        return info

