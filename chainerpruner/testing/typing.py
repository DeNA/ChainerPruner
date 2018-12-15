# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from typing import Union

import numpy as np
from chainer.backends import cuda

NdArray = Union[np.ndarray, cuda.ndarray]