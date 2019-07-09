# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import logging

import numpy as np

import chainerpruner
from chainerpruner import utils
from chainerpruner.rebuild.mapping import get_mapping

logger = logging.getLogger(__name__)


def reinitialize_model(model, mapping=None):

    if model.xp is not np:
        raise TypeError('please model.to_cpu()')

    if not mapping:
        mapping = get_mapping(model)

    options = dict()
    if utils.is_chainer_model(model):
        options['skipself'] = True

    for name, link in utils.named_modules(model, **options):
        logger.debug('name: {}'.format(name))
        rebuild_link = mapping[type(link)]()  # type: chainerpruner.rebuild.RebuildLink
        rebuild_link.apply_reinitialize(link)
