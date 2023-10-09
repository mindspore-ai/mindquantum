mindquantum.algorithm.compiler.euler_decompose
==============================================

.. py:function:: mindquantum.algorithm.compiler.euler_decompose(gate: QuantumGate, basis: str = 'zyz', with_phase: bool = True)

    单比特门欧拉分解。

    当前支持 `ZYZ` 和 `U3` 分解。

    参数：
        - **gate** (:class:`~.core.gates.QuantumGate`) - 一个单比特的量子门。
        - **basis** (str) - 分解的基，可以是 ``'zyz'`` 或者 ``'u3'`` 中的一个。默认值： ``'zyz'``。
        - **with_phase** (bool) - 是否将全局相位以 :class:`~.core.gates.GlobalPhase` 的形式作用在量子线路上。

    返回：
        List[:class:`~.core.circuit.Circuit`]，可能的分解方式。
