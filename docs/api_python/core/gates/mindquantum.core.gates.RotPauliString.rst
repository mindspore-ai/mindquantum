mindquantum.core.gates.RotPauliString
=====================================

.. py:class:: mindquantum.core.gates.RotPauliString(pauli_string: str, pr)

    任意泡利串的旋转门。

    .. math::

        U(\theta)=e^{-i\theta P/2}, P=\otimes_i\sigma_i, \text{where } \sigma \in \{X, Y, Z\}


    参数：
        - **pauli_string** (str) - 泡利串。泡利串中的元素只能是 ``['i', 'x', 'y', 'z', 'I', 'X', 'Y', 'Z']``。
        - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数。

    .. py:method:: diff_matrix(pr=None, about_what=None)

        返回该参数化量子门的导数矩阵。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 该参数化量子门的参数值。默认值：``None``。
            - **about_what** (str) - 关于哪个参数求导数。输入值为str类型的对应参数名。默认值：``None``。

        返回：
            numpy.ndarray，该量子门的导数矩阵形式。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。

    .. py:method:: matrix(pr=None, full=False)

        返回该参数化量子门的矩阵。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 该参数化量子门的参数值。默认值： ``None``。
            - **full** (bool) - 是否获取完整的矩阵（受控制比特和作用比特影响）。默认值： ``False``。

        返回：
            numpy.ndarray，该量子门的矩阵形式。
