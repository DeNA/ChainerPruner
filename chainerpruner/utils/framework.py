def is_chainer_model(model):
    import chainerpruner
    if not chainerpruner.avalable_chainer:
        return False

    import chainer
    if isinstance(model, chainer.Chain):
        return True
    else:
        return False


def is_pytorch_model(model):
    import chainerpruner
    if not chainerpruner.avalable_pytorch:
        return False

    import torch
    if isinstance(model, torch.nn.Module):
        return True
    else:
        return False


def named_modules(model, **options):
    if is_chainer_model(model):
        return model.namedlinks(**options)
    elif is_pytorch_model(model):
        return model.named_modules(**options)
    else:
        raise ModuleNotFoundError("{}".format(type(model)))
