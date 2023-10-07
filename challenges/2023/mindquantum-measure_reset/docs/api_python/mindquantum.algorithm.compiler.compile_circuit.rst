mindquantum.algorithm.compiler.compile_circuit
==============================================

.. py:function:: mindquantum.algorithm.compiler.compile_circuit(compiler: BasicCompilerRule, circ: Circuit)

    直接根据给定的编译规则，编译一个量子线路。

    参数：
        - **compiler** (:class:`~.algorithm.compiler.BasicCompilerRule`) - 编译规则。
        - **circ** (:class:`~.core.circuit.Circuit`) - 想要编译的量子线路。

    返回：
        :class:`~.core.circuit.Circuit`，编译后的量子线路。
