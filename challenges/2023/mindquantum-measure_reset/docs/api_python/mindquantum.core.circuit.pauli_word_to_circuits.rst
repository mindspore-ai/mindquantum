mindquantum.core.circuit.pauli_word_to_circuits
================================================

.. py:function:: mindquantum.core.circuit.pauli_word_to_circuits(qubitops)

    将单泡利词的量子算子转换成量子线路。

    参数：
        - **qubitops** (QubitOperator, Hamiltonian) - 单泡利词的量子算子。

    返回：
        Circuit，量子线路。

    异常：
        - **TypeError** - 如果 `qubitops` 不是 `Hamiltonian` 或 `QubitOperator` 。
        - **ValueError** - 如果 `qubitops` 是 `QubitOperator`，但不是在 ``'origin'`` 模式下。
        - **ValueError** - 如果 `qubitops` 有多个泡利词。
