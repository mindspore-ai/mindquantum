mindquantum.algorithm.compiler.cu_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.cu_decompose(gate: QuantumGate, with_barrier: bool = False)

    对任意维度的受控幺正门进行分解。

    该门拥有 :math:`m` 个控制比特和 :math:`n` 个作用比特。
    当迭代的调用函数本身时，:math:`m` 将会逐步减小并保持 :math:`n` 恒定。

    分解规则：

    - 当 :math:`m = 0`时，用 :func:`qs_decompose`
    - 当 :math:`m = 1`时，用 :func:`abc_decompose` 如果 :math:`n = 1`，用 :func:`qs_decompose` 如果 :math:`n > 1`。
    - 当 :math:`m > 1`时，当量子门时 Toffoli 门时，用 :func:`cxx_decompose`。其他情形，按照如下规则分解：

                 ─/──●───        ─/───────●──────────●────●──/─
                     │                    │          │    │
                 ────●───   ==   ────●────X────●─────X────┼────
                     │               │         │          │
                 ────U───        ────V─────────V†─────────V────

    这里V是U的矩阵的平方根。

    参数：
        - **gate** (:class:`QuantumGate`) - 量子门实例。
        - **with_barrier** (bool) - 是否在分解时加入 :class:`BarrierGate`。默认值： ``False``。

    返回：
        :class:`Circuit`，由单比特门和CNOT门构成的量子线路。
