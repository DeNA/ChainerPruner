__version__ = '0.1.0'
__copyright__ = 'Copyright (c) 2018 DeNA Co., Ltd.'
__license__ = 'MIT LICENSE'
__author__ = 'Tomohiro Kato'
__author_email__ = 'tomohiro.kato@dena.com'
__url__ = 'https://github.com/DeNA/ChainerPruner'

try:
    import chainer

    avalable_chainer = True
except ImportError:
    avalable_chainer = False

try:
    import torch

    avalable_pytorch = True
except ImportError:
    avalable_pytorch = False

import os
from chainerpruner import utils
from chainerpruner import masks
from chainerpruner import pruning
from chainerpruner import rebuild
from chainerpruner import serializers

from chainerpruner.pruner import Pruner
from chainerpruner.graph import Graph

disable_patch = os.getenv('CHAINERPRUNER_DISABLE_PYTORCH_LOAD_PATCH', False)

if avalable_pytorch and not disable_patch:
    from chainerpruner.serializers.pytorch import enable_custom_load_state_dict

    enable_custom_load_state_dict()
