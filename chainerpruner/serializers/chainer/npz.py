# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# Original code forked from MIT licensed chainer project
# https://github.com/chainer/chainer/blob/v5.1.0/chainer/serializers/npz.py

from logging import getLogger
import numpy

from chainer.backends import cuda
from chainer.backends import intel64
from chainer import serializers

logger = getLogger(__name__)


class NpzDeserializer(serializers.NpzDeserializer):

    """Deserializer for NPZ format.
    This is the standard deserializer in Chainer. This deserializer can be used
    to read an object serialized by :func:`save_npz`.

    Args:
        npz: `npz` file object.
        path: The base path that the deserialization starts from.
        strict (bool): If ``True``, the deserializer raises an error when an
            expected value is not found in the given NPZ file. Otherwise,
            it ignores the value and skip deserialization.
        ignore_names (string, callable or list of them):
            If callable, it is a function that takes a name of a parameter
            and a persistent and returns ``True`` when it needs to be skipped.
            If string, this is a name of a parameter or persistent that are
            going to be skipped.
            This can also be a list of callables and strings that behave as
            described above.
    """

    def __init__(self, npz, path='', strict=True, ignore_names=None):
        self.npz = npz
        self.path = path
        self.strict = strict
        if ignore_names is None:
            ignore_names = []
        self.ignore_names = ignore_names

    def __getitem__(self, key):
        key = key.strip('/')
        return NpzDeserializer(
            self.npz, self.path + key + '/', strict=self.strict,
            ignore_names=self.ignore_names)

    def __call__(self, key, value):
        key = self.path + key.lstrip('/')
        if not self.strict and key not in self.npz:
            return value

        if isinstance(self.ignore_names, (tuple, list)):
            ignore_names = self.ignore_names
        else:
            ignore_names = (self.ignore_names,)
        for ignore_name in ignore_names:
            if isinstance(ignore_name, str):
                if key == ignore_name:
                    return value
            elif callable(ignore_name):
                if ignore_name(key):
                    return value
            else:
                raise ValueError(
                    'ignore_names needs to be a callable, string or '
                    'list of them.')

        dataset = self.npz[key]
        if dataset[()] is None:
            return None

        if value is None:
            return dataset
        elif isinstance(value, numpy.ndarray):
            if value.shape != dataset.shape:
                # if shape of initialized weight shape and the shape of weight to load is difference,
                # initialized weight is resized to the shape of weight to load.
                logger.info('load %s: %s to %s', key, value.shape, dataset.shape)
                value.resize(dataset.shape, refcheck=False)
            numpy.copyto(value, dataset)
        elif isinstance(value, cuda.ndarray):
            # TODO(tkato) not supported
            value.set(numpy.asarray(dataset, dtype=value.dtype))
        elif isinstance(value, intel64.mdarray):
            # TODO(tkato) not supported
            intel64.ideep.basic_copyto(value, numpy.asarray(dataset))
        else:
            # TODO(tkato) not supported
            value = type(value)(numpy.asarray(dataset))
        return value


def load_npz(file, obj, path='', strict=True, ignore_names=None, verbose=False):
    """Loads an object from the file in NPZ format.
    This is a short-cut function to load from an `.npz` file that contains only
    one object.

    The only difference fron chainer.serializers.load_npz in that even if the
    model definition and the shape of weight to be loaded are different,
    it can be load.

    Args:
        file (str or file-like): File to be loaded.
        obj: Object to be deserialized. It must support serialization protocol.
        path (str): The path in the hierarchy of the serialized data under
            which the data is to be loaded. The default behavior (blank) will
            load all data under the root path.
        strict (bool): If ``True``, the deserializer raises an error when an
            expected value is not found in the given NPZ file. Otherwise,
            it ignores the value and skip deserialization.
        ignore_names (string, callable or list of them):
            If callable, it is a function that takes a name of a parameter
            and a persistent and returns ``True`` when it needs to be skipped.
            If string, this is a name of a parameter or persistent that are
            going to be skipped.
            This can also be a list of callables and strings that behave as
            described above.
        verbose (bool): If ``True``, it outputs verbose message on pruning statistics.
    .. seealso::
        :func:`chainer.serializers.save_npz`
    """
    with numpy.load(file) as f:
        d = NpzDeserializer(
            f, path=path, strict=strict, ignore_names=ignore_names)
        d.load(obj)

    # 読み込んだweightに合わせて各linkのattributeをupdateする
    from chainerpruner.rebuild.chainer.mapping import mapping
    for name, link in obj.namedlinks():
        rebuild_link_class = mapping.get(type(link),
                                         None)  # type: chainerpruner.rebuild.links.rebuildlink.RebuildLink
        if not rebuild_link_class:
            # 非サポートのOpはskip
            continue

        rebuild_link_class().update_attributes(link)
