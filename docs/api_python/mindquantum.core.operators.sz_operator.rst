mindquantum.core.operators.sz_operator
=======================================

.. py:function:: mindquantum.core.operators.sz_operator(n_spatial_orbitals)

    返回sz运算符。

    .. note::
        默认索引顺序自旋向上（α）对应于偶数索引，而自旋向下（β）对应于奇数索引。

    参数：
        - **n_spatial_orbitals** (int) - 空间轨道数（ `n_qubits // 2` ）。

    返回：
        `FermionOperator` ，对应于 `n_spatial_orbitals` 轨道上的sz运算符。
