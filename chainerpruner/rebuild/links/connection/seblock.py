# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import links as L

from chainerpruner.rebuild.links.rebuildlink import RebuildLink
from chainerpruner.rebuild.rebuild import passive_pruned_add


class RebuildSEBlock(RebuildLink):

    def passive_rebuild(self, se, mask):
        # in-hidden-inと戻すので、donw/up両方修正が必要
        se.down.W.array = se.down.W.array[:, mask].copy()
        se.up.W.array = se.up.W.array[mask, :].copy()
        se.up.b.array = se.up.b.array[mask].copy()

        # skipするため
        passive_pruned_add(se.down)
        passive_pruned_add(se.up)

    def reinitialize(self, se: L.Linear):
        raise NotImplementedError
