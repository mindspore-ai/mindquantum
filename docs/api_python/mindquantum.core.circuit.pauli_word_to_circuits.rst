.. py:function:: mindquantum.core.circuit.pauli_word_to_circuits(qubitops)

    将单Pauli词的量子算子转换成量子线路。

    **参数：**

    - **qubitops** (QubitOperator, Hamiltonian) - 单Pauli词的量子算子。

    **返回：**

    Circuit，量子线路。

    **异常：**

    - **TypeError** - 如果qubitops不是QubitOperator或QubitOperator。
    - **ValueError** - 如果量子点是QubitOperator的，但不是在origin模式下。
    - **ValueError** - 如果qubitops有多个Pauli词。
