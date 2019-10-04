# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import chainer
from chainer.training import extension

from chainerpruner.pruning.psfp.psfp import ProgressiveSoftFilterPruning


class ProgressiveSoftFilterPruningExtension(extension.Extension):
    name = 'ProgressiveSoftFilterPruning'
    trigger = (1, 'epoch')
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, args, target_layers,
                 pruning_rate, stop_trigger, pruning_rate_decay=1 / 8, rebuild=True):
        self._rebuild = rebuild
        self.core = ProgressiveSoftFilterPruning(
            model, args, target_layers,
            pruning_rate, stop_trigger, pruning_rate_decay)

    def __call__(self, trainer: chainer.training.Trainer):
        updater = trainer.updater
        epoch = updater.epoch
        iteration = updater.iteration

        step = iteration if self.trigger_type == 'iteration' else epoch
        self.core(step)

    def finalize(self):
        if self._rebuild:
            self.core.rebuild()
