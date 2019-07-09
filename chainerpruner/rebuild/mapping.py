# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainerpruner import utils


def get_mapping(model):
    if utils.is_chainer_model(model):
        from chainerpruner.rebuild.chainer.mapping import mapping
        return mapping
    elif utils.is_pytorch_model(model):
        from chainerpruner.rebuild.pytorch.mapping import mapping
        return mapping
    else:
        raise ModuleNotFoundError()
