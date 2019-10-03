# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import logging
from torch import nn

logger = logging.getLogger(__name__)


class Lasso():
    """Lasso regularization for PyTorch model."""

    def __init__(self, model, rate, target_bn_layers=None):
        """Lasso regularization for PyTorch model.

        Args:
            model (nn.Module):
            rate (float): Coefficient for Lasso.
            target_bn_layers (list[str]): Node name that apply Lasso.

        Returns:
            list[str]: Node names that actually applied Lasso.

        """
        self._model = model
        self._rate = rate
        self._target_layers = target_bn_layers

        if self._target_layers:
            count = 0
            for name, node in self._model.named_modules():
                if name in self._target_layers and isinstance(node, nn.BatchNorm2d):
                    count += 1
            if count == 0:
                raise AttributeError('target_layers={} does not exist in the model '
                                     'or does not BatchNorm2d'.format(self._target_layers))


    def __call__(self):
        info = list()
        for name, node in self._model.named_modules():
            if self._target_layers and name not in self._target_layers:
                continue
            if isinstance(node, nn.BatchNorm2d):
                info.append(name)
                node.weight.grad.data.add_(self._rate * node.weight.data.sign())
        return info

