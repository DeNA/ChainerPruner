import itertools

from torch.nn import Module
from torch.nn.parameter import Parameter


def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
    r"""Copies parameters and buffers from :attr:`state_dict` into only
    this module, but not its descendants. This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.

    Original code forked from PyTorch licensed PyTorch project:

    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py

    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.

    Arguments:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this moodule.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=False``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=False``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    for hook in self._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    local_name_params = list(itertools.chain(self._parameters.items(), self._buffers.items()))
    local_state = {k: v.data for k, v in local_name_params if v is not None}
    local_state_param = {k: v for k, v in local_name_params if v is not None}

    n_size_mismatch = 0

    for name, param in local_state.items():
        # weigt, bias, running_mean, running_varなどModuleのパラメータに対するloop
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]

            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if len(param.shape) == 0 and len(input_param.shape) == 1:
                input_param = input_param[0]

            if input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                # print('size mismatch for {}: copying a param with shape {} from checkpoint, '
                #       'the shape in current model is {}.'
                #       .format(key, input_param.shape, param.shape))
                # PyTorch default implementation is continue
                # continue
                n_size_mismatch += 1

            if isinstance(input_param, Parameter):
                # backwards compatibility for serialized parameters
                input_param = input_param.data
            try:
                # 強制的にロード
                # param.copy_(input_param)
                local_state_param[name].data = input_param
            except Exception:
                error_msgs.append('While copying the parameter named "{}", '
                                  'whose dimensions in the model are {} and '
                                  'whose dimensions in the checkpoint are {}.'
                                  .format(key, param.size(), input_param.size()))
        elif strict:
            missing_keys.append(key)

    if strict:
        for key, input_param in state_dict.items():
            if key.startswith(prefix):
                input_name = key[len(prefix):]
                input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                if input_name not in self._modules and input_name not in local_state:
                    unexpected_keys.append(key)

    if local_state and n_size_mismatch > 0:
        # update attributes
        from chainerpruner.rebuild.pytorch.mapping import mapping
        # print('update property: ', self)
        rebuild_link = mapping.get(type(self), None)  # type: chainerpruner.rebuild.RebuildLink
        if rebuild_link:
            rebuild_link = rebuild_link()
            rebuild_link.update_attributes(self)


def enable_custom_load_state_dict():
    """Module#load_state_dictにパッチを当ててpruning後のパラメータをロードできるようにする

    - モデル定義とパラメータのサイズが異なる場合でも強制的にロードする
    - パラメータのサイズに依存するModuleのプロパティを変更する

    Example
    -------

        model = MobileNetV2()

        # PyTorchの標準ではモデル定義と異なるサイズのモデルはloadできないため、パッチを当てる
        # model.load_state_dictの前であればどこで呼び出してもよい
        enable_custom_load_state_dict()

        model.load_state_dict(torch.load(PATH))

    """
    Module._load_from_state_dict = _load_from_state_dict
