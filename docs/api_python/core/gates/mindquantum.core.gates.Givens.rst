mindquantum.core.gates.Givens
=================================

.. py:class:: mindquantum.core.gates.Givens(pr)

    Givens 门。更多用法，请参见 :class:`~.core.gates.RX`。

    .. math::

        {\rm G}(\theta)=\exp{\left(-i\frac{\theta}{2} (Y\otimes X - X\otimes Y)\right)} =
        \begin{pmatrix}
            1 & 0 & 0 & 0\\
            0 & \cos{\theta} & -\sin{\theta} & 0\\
            0 & \sin{\theta} & \cos{\theta} & 0\\
            0 & 0 & 0 & 1\\
        \end{pmatrix}

    参数：
        - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: diff_matrix(pr=None, about_what=None)

        返回该参数化量子门的导数矩阵。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 该参数化量子门的参数值。默认值： ``None``。
            - **about_what** (str) - 关于哪个参数求导数。输入值为str类型的对应参数名。默认值： ``None``。

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
