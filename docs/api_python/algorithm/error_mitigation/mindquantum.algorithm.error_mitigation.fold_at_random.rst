mindquantum.algorithm.error_mitigation.fold_at_random
=====================================================

.. py:function:: mindquantum.algorithm.error_mitigation.fold_at_random(circ: Circuit, factor: float, method='locally')

    随机的折叠一个量子线路。

    折叠一个量子线路会增加量子线路的长度，但是仍然保持量子线路的幺正矩阵不变。我们可以通过在某些量子门后添加一个单位量子线路来实现。举一个简单的例子，:math:`RX(1.2 \pi)` 和 :math:`RX(1.2 \pi)RX(-1.2 \pi)RX(1.2 \pi)` 拥有相同的幺正矩阵表示，但是后者的线路长度却增加了三倍。

    参数：
        - **circ** (:class:`~.core.circuit.Circuit`) - 要折叠的量子线路。
        - **factor** (float) - 折叠系数，必须大于1。
        - **method** (str) - 折叠的方法。 ``method`` 应该是 ``'globally'`` 或者 ``'locally'`` 中的一种。其中 ``'globally'`` 方法表示在整个量子线路后面增加单位量子线路，而 ``'locally'`` 表示随机的在某些量子门后面添加单位量子线路。
