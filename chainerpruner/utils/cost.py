# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import chainer

try:
    from chainer_computational_cost import ComputationalCostHook
    enable_chainer_computational_cost = True
except:
    enable_chainer_computational_cost = False

from chainerpruner.rebuild.links.mapping import mapping


def calc_computational_cost(model: chainer.Chain, args, fma_1flop=True, custom_calculators=None):
    """calculation computational cost using chainer_computational_cost

    https://github.com/belltailjp/chainer_computational_cost

    Args:
        model:
        args:
        custom_calculators:

    Returns:

    """

    if not enable_chainer_computational_cost:
        raise ImportError()

    # get custom calculator
    if custom_calculators is None:
        custom_calculators = dict()
        for link, rebuild_link in mapping.items():
            custom_calculator = rebuild_link().get_computational_cost_custom_calculator()
            if custom_calculator is not None:
                custom_calculators[link] = custom_calculator

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        with ComputationalCostHook(fma_1flop=fma_1flop) as cch:
            for link, custom_calculator in custom_calculators.items():
                cch.add_custom_cost_calculator(link, custom_calculator)
            model(args)

    return cch


