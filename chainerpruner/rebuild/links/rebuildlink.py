# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import logging

import chainer

from chainerpruner.testing.typing import NdArray

logger = logging.getLogger(__name__)


class RebuildLink():

    def __init__(self):
        self.logger = logger
        self.node = None
        self.custom_calculator = None

    def apply_active_rebuild(self, link: chainer.Link) -> NdArray:
        mask = self.active_rebuild(link)
        if mask is None:
            raise RuntimeError('active_rebuild() return None, please check implementation.')

        if not mask.any():
            logger.warning('mask is all False. not pruning')

        return mask

    def apply_passive_rebuild(self, link: chainer.Link, mask: NdArray):
        if not mask.any():
            logger.warning('mask is all False. not pruning')
        self.passive_rebuild(link, mask)

    def apply_reinitialize(self, link: chainer.Link):
        try:
            self.reinitialize(link)
        except NotImplementedError as e:
            logger.warning('use reinitialize default implementation: {}, {}'.fomrat(link.name, type(link)))
            # default implementation
            # variable.Parameter::initializeを呼び出し、old_paramのshapeでinitializeする
            for name, param in link.namedparams():
                param.initialize(param.shape)

    def active_rebuild(self, link: chainer.Link):
        """zeroのchannelを除外してweightを再構築する

        Args:
            link:

        Returns:

        """
        raise NotImplementedError()

    def passive_rebuild(self, link: chainer.Link, mask: NdArray):
        """

        Args:
            link:
            mask:

        Returns:

        """
        raise NotImplementedError()

    def reinitialize(self, link: chainer.Link):
        """

        Args:
            link:

        Returns:

        """
        raise NotImplementedError

    def get_computational_cost_custom_calculator(self):
        if self.custom_calculator is not None:
            return self.custom_calculator
        else:
            return None
