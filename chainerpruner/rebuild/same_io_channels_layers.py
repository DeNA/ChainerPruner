# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import links as L


# for example
# conv1(in:3, out:32)-bn1(in/out: 32)-conv2(in: 32, out:64)
# conv1 pruned (32->8) then bn1 and conv2
# conv1(in:3, out:16)-bn1(in/out: 16)-conv2(in: 16, out:64)
# TODO(tkato) conv2d(group=out_channels)
same_io_channels_layers = [
    L.DepthwiseConvolution2D,
    L.BatchNormalization,
]
