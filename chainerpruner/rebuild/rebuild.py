# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import logging

from chainerpruner.rebuild.calc_pruning_connection import calc_pruning_connection

logger = logging.getLogger(__name__)

__passive_pruned = set()


def passive_pruned_add(node):
    global __passive_pruned
    __passive_pruned.add(node)


def passive_pruned_clear():
    global __passive_pruned
    __passive_pruned.clear()


def rebuild(model, graph, target_layers, mapping=None):
    """rebuild each weight

    Args:
        model:
        graph:
        target_layers:
        mapping:

    Returns:

    """
    passive_pruned_clear()

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

    nodes = {node.name: node for node in graph.graph.nodes}

    pruned = set()
    count = 0
    for name, post_names in pruning_connection_info.items():
        if name not in target_layers:
            continue
        logger.debug('(active)%s: %s', name, post_names)

        # rebuild pruning target node
        target_link = model_dict[name]
        rebuild_link_class = mapping.get(type(target_link),
                                         None)  # type: chainerpruner.rebuild.links.rebuildlink.RebuildLink
        if rebuild_link_class is None:
            raise NotImplementedError
        rebuild_link = rebuild_link_class()
        rebuild_link.node = nodes[name]
        mask = rebuild_link.apply_active_rebuild(target_link)

        info.append({
            'name': name,
            'before': len(mask),
            'after': int(sum(mask)),
        })

        # later node rebuild (input channels)
        for post_name in post_names:
            logger.debug('(passive)%s:', post_name)
            if post_name in pruned:
                continue

            target_link = model_dict[post_name]

            # passive rebuild済のノードはskipする
            # 例えばSEBlock(Linear, Linear)のようにUserDefinedChainのまとまりとして
            # in/outチャネルの整合性を保つ必要がある層がある
            # この場合、SEBlockのpassive rebuildのクラスをテーブルに追加しておき、
            # SEBlockを構成するLinearのpassive rebuildはskipするようにする
            if target_link in __passive_pruned:
                continue

            rebuild_link_class = mapping.get(type(target_link), None)
            if rebuild_link_class is None:
                # ResBlockなどUserDefinedLinkを含む場合があるのでskip
                continue

            rebuild_link = rebuild_link_class()
            rebuild_link.node = nodes[post_name]
            rebuild_link.apply_passive_rebuild(target_link, mask.copy())
            pruned.add(post_name)

        count += 1

    if count == 0:
        logger.warning('rebuild layer not found')

    passive_pruned_clear()
    return info
