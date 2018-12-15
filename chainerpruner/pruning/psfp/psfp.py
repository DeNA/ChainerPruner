# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import logging

import numpy as np

try:
    from scipy.optimize import curve_fit
    enable_scipy = True
except:
    enable_scipy = False

import chainer
from chainer.training import extension

from chainerpruner import Graph
from chainerpruner.masks import NormMask
from chainerpruner.rebuild.rebuild import rebuild

logger = logging.getLogger(__name__)


class ProgressiveSoftFilterPruning(extension.Extension):

    name = 'ProgressiveSoftFilterPruning'
    trigger = (1, 'epoch')
    priority = chainer.training.PRIORITY_WRITER # TODO(tkato)

    def __init__(self, model, args, target_layers,
                 pruning_rate, stop_trigger, pruning_rate_decay=1 / 8, rebuild=True):
        """ Progressive Deep Neural Networks Acceleration via Soft Filter Pruning

        https://arxiv.org/abs/1808.07471

        Args:
            model (chainer.Chain):
            target_layers (list):
            pruning_rate (float): sparsity. target_layerで指定した全レイヤ一律 [0, 1) 大きいほど高圧縮
            pruning_rate_decay (float): pruning_rateのprogressiveな変化率を調整するパラメータ。論文では1/8がデフォルト
                pruning_rateの3/4のsparsityを学習のmax_iteration/epochの何%の位置に指定するか
            trigger (tuple): weightをzeroにする頻度 (500, 'iteration') のように指定する。論文では(1, 'epoch')がデフォルト
            stop_trigger (tuple): 学習の総iteration/epochを指定。 triggerとstop_triggerの単位はiterationかepochで合わせること
        """

        if not enable_scipy:
            raise ImportError() # TODO(tkato)

        self.model = model
        self.target_layers = target_layers
        self.pruning_rate = pruning_rate
        self.pruning_rate_decay = pruning_rate_decay
        self.trigger_type = self.trigger[1]
        self.stop_trigger = stop_trigger
        self.rebuild = rebuild

        self.graph = Graph(model, args)

        initial_pruning_rate = 0.
        self.mask = NormMask(model, self.graph, target_layers, percent=initial_pruning_rate, norm='l2')

        assert self.trigger_type == stop_trigger[1], 'extensions trigger == stop trigger'

        self._pruning_rate_fn = self._init_pruning_rate_fn(pruning_rate,
                                                           pruning_rate_decay,
                                                           stop_trigger[0])

    def _init_pruning_rate_fn(self, pruning_rate, pruning_rate_decay, max_step):
        """progressiveにpruning ratioを上昇させる関数を構築

        curve-fitting to y = a * exp(-k * x) + b
        (0, 0), (max_step * pruning_rate_decay, pruning_rate / 4), (max_step, pruning_rate)

        Args:
            pruning_rate:
            pruning_rate_decay:
            max_step:

        Returns:
            fn: callable
        """
        pruning_rate *= 100

        def f(x, a, k, b):
            return a * np.exp(-k * x) + b

        # using fp64
        xdata = np.array([0, max_step * pruning_rate_decay, max_step], dtype=np.float64)
        ydata = np.array([0, pruning_rate * 3 / 4, pruning_rate], dtype=np.float64)  # paper = 1/4 ?
        p0 = np.array([0, 0, 0], dtype=np.float32)
        popt, _ = curve_fit(f, xdata, ydata, p0=p0)

        print('({}, sparsity[%]): {}'.format(self.trigger_type, [(x, y) for x, y in zip(xdata, ydata)]))

        return lambda x: f(x, *popt) * 0.01  # 10% -> 0.1

    def __call__(self, trainer: chainer.training.Trainer):

        updater = trainer.updater
        epoch = updater.epoch
        iteration = updater.iteration
        step = iteration if self.trigger_type == 'iteration' else epoch

        # update pruning_rate
        for key in self.mask.percent.keys():
            self.mask.percent[key] = self._pruning_rate_fn(step)

        info = self.mask()

    def finalize(self):
        if self.rebuild:
            info = rebuild(self.model, self.graph, self.target_layers)
            logger.debug(info)
