# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import torch.nn as nn

from chainerpruner.rebuild.pytorch.modules.batchnorm import RebuildBatchNormalization
from chainerpruner.rebuild.pytorch.modules.conv import RebuildConvolution2D
from chainerpruner.rebuild.pytorch.modules.linear import RebuildLinear

mapping = {
    nn.Conv2d: RebuildConvolution2D,
    nn.BatchNorm2d: RebuildBatchNormalization,
    nn.Linear: RebuildLinear,
}