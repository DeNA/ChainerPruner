# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from collections import defaultdict
from typing import Optional, Union, Mapping

from chainerpruner.mask import Mask


def set_all_layers(target_layers, threshold_or_percent, default):
    if isinstance(threshold_or_percent, dict):
        if set(target_layers) != set(threshold_or_percent.keys()) and default is None:
            raise ValueError
        ret = defaultdict(lambda: default)
        ret.update(threshold_or_percent)
    elif isinstance(threshold_or_percent, float):  # TODO(tkato) numeric
        if default is not None:
            raise ValueError
        ret = {name: threshold_or_percent for name in target_layers}
    else:
        raise TypeError()
    return ret


class NormMask(Mask):

    def __init__(self, model, graph, target_layers,
                 threshold: Optional[Union[float, Mapping[str, float]]] = None,
                 percent: Optional[Union[float, Mapping[str, float]]] = None,
                 default: Optional[float] = None,
                 mask_layer=None,
                 norm='l1'):
        super(NormMask, self).__init__(model, graph, target_layers,
                                       mask_layer=mask_layer)

        self.model = model
        self.norm = norm

        # TODO(tkato) check model init

        self.threshold = None
        self.percent = None
        if threshold and percent:
            raise AttributeError('threshold or percent') # TODO(tkato)
        elif threshold is not None:
            self.threshold = set_all_layers(target_layers, threshold, default)
        elif percent is not None:
            self.percent = set_all_layers(target_layers, percent, default)
        else:
            raise AttributeError()

    def get_filter_norm(self, mask):
        if self.is_chainer():
            return self._get_filter_norm_chainer(mask)
        else:
            return self._get_filter_norm_pytorch(mask)

    def _get_filter_norm_chainer(self, mask):
        from chainer import backend
        xp = backend.get_array_module(mask)

        if self.norm == 'l1':
            def l1norm(p):
                return xp.sum(p, axis=(1, 2, 3), keepdims=True)
            mask = l1norm(mask)
        elif self.norm == 'l2':
            def l2norm(p):
                return xp.sqrt(xp.sum(xp.square(p), axis=(1, 2, 3), keepdims=True))
            mask = l2norm(mask)
        else:
            raise NotImplementedError()
        return mask

    def _get_filter_norm_pytorch(self, mask):
        import torch
        if self.norm == 'l1':
            def l1norm(p):
                return torch.sum(p, dim=(1, 2, 3), keepdim=True)
            mask = l1norm(mask)
        elif self.norm == 'l2':
            def l2norm(p):
                return torch.sqrt(torch.sum(p ** 2, dim=(1, 2, 3), keepdim=True))
            mask = l2norm(mask)
        else:
            raise NotImplementedError()
        return mask

    def get_thresholds(self, name, mask):
        if self.is_chainer():
            return self._get_thresholds_chainer(name, mask)
        else:
            return self._get_thresholds_pytorch(name, mask)

    def _get_thresholds_chainer(self, name, mask):
        """

        Args:
            name:
            mask (numpy.ndarray or cupy.ndarray):

        Returns:

        """
        from chainer import backend
        xp = backend.get_array_module(mask)

        if self.threshold is not None:
            thresh = self.threshold[name]
            return thresh
        elif self.percent is not None:
            # percent to threshold
            thresh_index = int(self.percent[name] * mask.size)
            thresh = float(xp.sort(mask.flatten())[thresh_index])
            return thresh
        else:
            raise ValueError()

    def _get_thresholds_pytorch(self, name, mask):
        """

        Args:
            name:
            mask (torch.Tensor):

        Returns:

        """
        if self.threshold is not None:
            thresh = self.threshold[name]
            return thresh
        elif self.percent is not None:
            # percent to threshold
            thresh_index = int(self.percent[name] * mask.numel())
            thresh = float(mask.flatten().sort()[0][thresh_index])
            return thresh
        else:
            raise ValueError()
