# Copyright (c) 2018 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

from chainer import links as L

from chainerpruner.rebuild.rebuildlink import RebuildLink
from chainerpruner.rebuild.rebuild import passive_pruned_add


class RebuildSEBlock(RebuildLink):

    def update_attributes(self, se):
        se.down.out_size, se.down.in_size = se.down.W.shape
        se.up.out_size, se.up.in_size = se.up.W.shape

    def passive_rebuild(self, se, mask):
        # in-hidden-inと戻すので、donw/up両方修正が必要
        se.down.W.array = se.down.W.array[:, mask].copy()
        se.up.W.array = se.up.W.array[mask, :].copy()
        se.up.b.array = se.up.b.array[mask].copy()

        # seblock内のlinkをrebuild済とし、これ以降rebuildされないようにする
        #
        # 現状の実装では、seblockとseblock/upとseblock/downそれぞれrebuildされる
        # 通常はseblockのようなUserDefinedChainはskipされプリミティブなup/down（L.Linear）のみ
        # rebuildされる。しかしSEBlockはin-hidden-inとチャネル数が定義されているため、
        # up/downを単独にrebuildするのではなく、このクラスの中でseblockとしてrebuildしChain内の
        # チャネルの整合性を調整する。
        # このような例外に対応するためにpassive_pruned_addを定義しており、指定したlinkのrebuildを
        # skipすることができる
        passive_pruned_add(se.down)
        passive_pruned_add(se.up)

    def reinitialize(self, se: L.Linear):
        # TODO(tkat0)
        raise NotImplementedError
