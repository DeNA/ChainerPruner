# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import links as L
from chainerpruner.rebuild.chainer.links.normalization.batch_normalization import RebuildBatchNormalization
from chainerpruner.rebuild.chainer.links.connection.convolution_2d import RebuildConvolution2D
from chainerpruner.rebuild.chainer.links.connection.depthwise_convolution_2d import RebuildDepthwiseConvolution2D
from chainerpruner.rebuild.chainer.links.connection.linear import RebuildLinear
from chainerpruner.rebuild.chainer.links.connection.seblock import RebuildSEBlock
import chainercv

# Channel Pruningに対応するLink
mapping = {
    L.Convolution2D: RebuildConvolution2D,
    L.DilatedConvolution2D: RebuildConvolution2D,
    L.DepthwiseConvolution2D: RebuildDepthwiseConvolution2D,
    L.BatchNormalization: RebuildBatchNormalization,
    L.Linear: RebuildLinear,
    chainercv.links.connection.seblock.SEBlock: RebuildSEBlock,
}
