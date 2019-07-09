# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import logging

logger = logging.getLogger(__name__)


class RebuildLink():
    """Channel PruningによるLinkの変更など、Link毎の実装をするためのベースクラス"""

    def __init__(self):
        self.logger = logger
        self.node = None
        self.custom_calculator = None

    def apply_update_attributes(self, link):
        """out_channelsのようなLinkのattributeをpruning後に修正する"""
        # TODO(tkat0) 各メソッドの後直接呼んでいるが、適切な終了処理に押し込みたい
        self.update_attributes(link)

    def apply_active_rebuild(self, link):
        mask = self.active_rebuild(link)
        if mask is None:
            raise RuntimeError('active_rebuild() return None, please check implementation.')

        if not mask.any():
            logger.warning('mask is all False. not pruning')

        self.apply_update_attributes(link)

        return mask

    def apply_passive_rebuild(self, link, mask):

        try:
            mask = mask.copy()  # xp.ndarray
        except AttributeError:
            mask = mask.clone()  # torch.Tensor

        if not mask.any():
            logger.warning('mask is all False. not pruning')
        self.passive_rebuild(link, mask)

        self.apply_update_attributes(link)

    def apply_reinitialize(self, link):
        try:
            self.reinitialize(link)
        except NotImplementedError as e:
            logger.warning('use reinitialize default implementation: {}, {}'.fomrat(link.name, type(link)))
            # TODO(tkat0) pytorch
            # default implementation
            # variable.Parameter::initializeを呼び出し、old_paramのshapeでinitializeする
            for name, param in link.namedparams():
                param.initialize(param.shape)

        self.apply_update_attributes(link)

    def update_attributes(self, link):
        """rebuildにより変更になったLinkのattributeをupdateする

        例えばout_channels。このattributeを見ているモジュールもいるため修正する必要がある

        Args:
            link:

        Returns:

        """
        raise NotImplementedError()

    def active_rebuild(self, link):
        """zeroのchannelを除外してweightを再構築する

        Args:
            link:

        Returns:

        """
        raise NotImplementedError()

    def passive_rebuild(self, link, mask):
        """1つ前の層のmaskから、自身の層の入力チャネルをpruningしweightを再構築する

        Args:
            link:
            mask:

        Returns:

        """
        raise NotImplementedError()

    def reinitialize(self, link):
        """rebuild後のshapeが変更されたweightに対して、
        設定されたinitializersを用いて再初期化する

        rebuild後にscratchで学習するケースで利用

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
