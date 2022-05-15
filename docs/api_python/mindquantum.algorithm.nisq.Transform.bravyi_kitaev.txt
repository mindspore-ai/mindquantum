mindquantum.algorithm.nisq.Transform.bravyi_kitaev()

        应用Bravyi-Kitaev变换。

        布拉维-基塔耶夫基础是乔丹-威格纳和奇偶校验转换之间的中间。
        也就是说，它平衡了占用的局部性和奇偶校验信息，以提高模拟效率。
        在此方案中，量子位存储一组 :math: `2^x`轨道的奇偶性，其中 :math: `x\ge 0`。索引j的量子位总是存储轨道 :math: `j`。
        对于 :math: `j`的偶数值，这是它存储的唯一轨道，但对于 :math: `j`的奇数值，它还存储索引小于 :math: `j`的一组相邻轨道。
        对于职业转变，我们遵循公式：

        .. math::

            b_{i} = \sum{[\beta_{n}]_{i,j}} f_{j},

        其中 :math: '\β_{n}'是 :math: `N\乘以N`平方矩阵， :math: `N`是总量子位数。
        量子位索引分为三个集合，奇偶校验集、更新集和翻转集。
        这组量子位的奇偶校验与索引小于 :math: `j`的轨道集具有相同的奇偶校验，因此我们将将这组量子位索引称为索引的“奇偶校验集” :math: `j`，或 :math: 'P(j)'。

        索引的更新集 :math: `j`，或 :math: `U(j)`包含量子位（量子位除外 :math: `j`），当占用轨道：`j`时，必须更新这些量子位（量子位除外 :math: `j`）
        这是Bravyi-Kitaev基中的量子位集，它存储了包括轨道在内的部分和 :math: `j`。
        索引的翻转集 :math: `j`，或 :math: `F(j)'包含BravyiKitaev量子位集，确定量子位 :math: `j`相对于轨道是否具有相同的奇偶校验或倒奇偶校验 :math: j。

        请参见论文中的一些详细解释 (THE JOURNAL OF CHEMICAL PHYSICS 137, 224109 (2012)).

        从https://arxiv.org/pdf/quant-ph/0003137.pdf 和 Peter M.Fenwick的“累积频率表的新数据结构”实现。

        返回:
            QubitOperator，布拉维伊_基塔耶夫转换后的量子位运算符。
        