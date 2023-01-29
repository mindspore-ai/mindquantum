mindquantum.core.operators.Projector
=====================================

.. py:class:: mindquantum.core.operators.Projector(proj)

    投影算子。

    对于一个如下所示的投影算子：

    .. math::

        \left|01\right>\left<01\right|\otimes I^2

    字符串格式为'01II'。

    .. note::
        索引小的量子比特位于bra和ket字符串格式的右端。

    参数：
        - **proj** (str) - 投影算子的字符串格式。
