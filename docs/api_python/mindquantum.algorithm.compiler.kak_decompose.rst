mindquantum.algorithm.compiler.kak_decompose
============================================

.. py:function:: mindquantum.algorithm.compiler.kak_decompose(gate: QuantumGate, return_u3: bool = True)

    通过kak分解来分解任意的两量子比特门。

    第一步：先按照如下形式进行分解

             ┌──────────┐
        ──B0─┤          ├─A0──
             │ exp(-iH) │
        ──B1─┤          ├─A1──
             └──────────┘
    .. math::

        \left( A_0 \otimes A_1 \right) e^{-iH}\left( B_0 \otimes B_1 \right)

    第二部：利用三个CNOT门来分解e指数部分。

        ──B0────●────U0────●────V0────●────W─────A0──
                │          │          │
        ──B1────X────U1────X────V1────X────W†────A1──

    更多信息，请参考论文 `An Introduction to Cartan's KAK Decomposition for QC
    Programmers <https://arxiv.org/abs/quant-ph/0406176>`_.

    参数：
        - **gate** (:class:`QuantumGate`) - 只有一个控制为的单比特量子门。
        - **return_u3** (bool) - 如果为 ``True``，则返回 :class:`U3` 形式的分解，否则返回 :class:`UnivMathGate` 形式的分解。默认值： ``True``。

    返回：
        :class:`Circuit`，由6个单比特门和最多三个CNOT门构成的量子线路。
