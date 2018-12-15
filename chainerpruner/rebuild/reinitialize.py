# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import logging

import numpy as np
import chainer

import chainerpruner

logger = logging.getLogger(__name__)


def reinitialize_model(model: chainer.Chain, mapping=None):

    if model.xp is not np:
        raise TypeError('please model.to_cpu()')

    if not mapping:
        from chainerpruner.rebuild.links.mapping import mapping as m
        mapping = m

    for name, link in model.namedlinks(skipself=True):
        logger.debug('name: {}'.format(name))
        rebuild_link = mapping[type(link)]()  # type: chainerpruner.rebuild.links.rebuildlink.RebuildLink
        rebuild_link.apply_reinitialize(link)
