.. py:method:: mindquantum.core.operators.FermionOperator.matrix(n_qubits=None)

    将此费米子运算符转换为jordan_wigner映射下的csr_matrix。

    **参数：**

    - **n_qubits** (int) - 结果矩阵的总量子位。如果是None，则该值将是最大本地量子位数。默认值：None。
    