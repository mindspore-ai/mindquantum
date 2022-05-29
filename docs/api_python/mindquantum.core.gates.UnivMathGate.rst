.. py:class:: mindquantum.core.gates.UnivMathGate(name, matrix_value)

    通用数学门。

    更多用法，请参见 :class:`mindquantum.core.gates.XGate`.

    **参数：**
    - **name** (str) - 此门的名称。
    - **mat** (np.ndarray) - 此门的矩阵值。

    **样例：**
        >>> from mindquantum.core.gates import UnivMathGate
        >>> x_mat=np.array([[0,1],[1,0]])
        >>> X_gate=UnivMathGate('X',x_mat)
        >>> x1=X_gate.on(0,1)
        >>> print(x1)
        X(0 <-: 1)
    