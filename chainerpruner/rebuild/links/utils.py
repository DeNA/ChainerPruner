# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]


def log_shape(weight, mask):
    return 'weight: {}, mask: {}'.format(weight.shape, mask.shape)
