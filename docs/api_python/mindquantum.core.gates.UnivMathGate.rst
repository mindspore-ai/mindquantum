.. py:class:: mindquantum.core.gates.UnivMathGate(name, matrix_value)

    通用数学门。

    更多用法，请参见 :class:`mindquantum.core.gates.XGate`.

    **参数：**

    - **name** (str) - 此门的名称。
    - **mat** (np.ndarray) - 此门的矩阵值。

    .. py:method:: get_cpp_obj()

        返回地层c++对象。
