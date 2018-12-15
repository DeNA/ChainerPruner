# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import logging

from chainerpruner.rebuild.calc_pruning_connection import calc_pruning_connection

logger = logging.getLogger(__name__)


def rebuild(model, graph, target_layers, mapping=None):
    """rebuild each weight

    Args:
        model:
        graph:
        target_layers:
        mapping:

    Returns:

    """

    if not mapping:
        from chainerpruner.rebuild.links.mapping import mapping as m
        mapping = m

    if len(target_layers) == 0:
        raise ValueError('invalid rebuild_info')

    pruning_connection_info = calc_pruning_connection(graph)
    if not pruning_connection_info:
        raise ValueError('pruinng_connection_info parse error')
    logger.debug('pruning_connection_info', pruning_connection_info)

    model_dict = {name: link for name, link in model.namedlinks()}
    info = []

    count = 0
    for name, post_names in pruning_connection_info.items():
        if name not in target_layers:
            continue
        logger.debug('(active)%s: %s', name, post_names)

        # rebuild pruning target node
        target_link = model_dict[name]
        rebuild_link = mapping[type(target_link)]()  # type: chainerpruner.rebuild.links.rebuildlink.RebuildLink
        rebuild_link.node = graph.links[name]
        mask = rebuild_link.apply_active_rebuild(target_link)

        info.append({
            'name': name,
            'before': len(mask),
            'after': int(sum(mask)),
        })

        # later node rebuild (input channels)
        for post_name in post_names:
            logger.debug('(passive)%s:', post_name)
            target_link = model_dict[post_name]
            rebuild_link = mapping[type(target_link)]()
            rebuild_link.node = graph.links[post_name]
            rebuild_link.apply_passive_rebuild(target_link, mask.copy())

        count += 1

    if count == 0:
        logger.warning('rebuild layer not found')

    return info
