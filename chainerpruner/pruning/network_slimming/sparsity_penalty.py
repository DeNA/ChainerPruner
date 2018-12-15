# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import cuda

import logging

# https://github.com/chainer/chainer/blob/v5.1.0/chainer/optimizer_hooks/lasso.py

from chainer import backend
from chainer import cuda

logger = logging.getLogger(__name__)


class Lasso(object):
    """Optimizer/UpdateRule hook function for Lasso regularization.
    This hook function adds a scaled parameter to the sign of each weight.
    It can be used as a regularization.
    Args:
        rate (float): Coefficient for the weight decay.
    Attributes:
        ~optimizer_hooks.Lasso.rate (float): Coefficient for the weight decay.
        ~optimizer_hooks.Lasso.timing (string): Specifies
                         when this hook should be called by
                         the Optimizer/UpdateRule. Valid values are 'pre'
                         (before any updates) and 'post' (after any updates).
        ~optimizer_hooks.Lasso.call_for_each_param (bool): Specifies
                         if this hook is called for each parameter (``True``)
                         or only once (``False``) by an optimizer to
                         which this hook is registered. This function does
                         not expect users to switch the value from default one,
                         which is `True`.
    .. versionadded:: 4.0.0
       The *timing* parameter.

    違いは、target_layersで指定したbn.gammaにのみかけること
    """
    name = 'Lasso'
    call_for_each_param = False
    timing = 'pre'

    def __init__(self, rate, target_batchnorm_layers=None):
        self.rate = rate
        self.target_layers = target_batchnorm_layers

    def __call__(self, opt):
        count = 0
        for name, param in opt.target.namedparams():
            p, g = param.data, param.grad
            if p is None or g is None:
                return

            # gammaを含むLink(BatchNormalization)がpruning対象
            # targets指定がある場合は、その名前を含むBatchNormalizationのみ対象
            # targetsがNoneなら、すべてのBatchNormalizationが対象
            if 'gamma' not in name:
                continue
            elif self.target_layers is not None and not any([target in name for target in self.target_layers]):
                continue
            else:
                # pruning targets
                count += 1
                logger.debug('Lasso: {}'.format(name))
                pass

            # TODO(tkato) reporter

            xp = backend.get_array_module(p)
            with cuda.get_device_from_array(p) as dev:
                sign = xp.sign(p)
                if int(dev) == -1:
                    g += self.rate * sign
                else:
                    kernel = cuda.elementwise(
                        'T s, T decay', 'T g', 'g += decay * s', 'lasso')
                    kernel(sign, self.rate, g)
        if count == 0:
            logger.warning('Lasso is not apply')
