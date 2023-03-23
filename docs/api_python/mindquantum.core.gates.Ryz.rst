mindquantum.core.gates.Ryz
===============================

.. py:class:: mindquantum.core.gates.Ryz(pr)

    Ryz 门。更多用法，请参见 :class:`mindquantum.core.gates.RX`。

    .. math::

        {\rm Ryz_\theta}=\exp{-i\frac{\theta}{2} Z\otimes Y} =\begin{pmatrix}
            \cos{\frac{\theta}{2}} & -\sin{\frac{\theta}{2}} & 0 & 0\\
            \sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}} & 0 & 0\\
            0 & 0 & \cos{\frac{\theta}{2}} & \sin{\frac{\theta}{2}}\\
            0 & 0 & -\sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}}\\
            \end{pmatrix}

    参数：
        - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: diff_matrix(pr=None, about_what=None, frac=0.5)

        返回该参数化量子门的导数矩阵。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 该参数化量子门的参数值。默认值：None。
            - **about_what** (str) - 关于哪个参数求导数。输入值为str类型的对应参数名。默认值：None。
            - **frac** (numbers.Number) - 系数的倍数。默认值：0.5。

        返回：
            numpy.ndarray，该量子门的导数矩阵形式。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。

    .. py:method:: matrix(pr=None, frac=0.5)

        返回该参数化量子门的矩阵。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 该参数化量子门的参数值。默认值：None。
            - **frac** (numbers.Number) - 系数的倍数。默认值：0.5。

        返回：
            numpy.ndarray，该量子门的矩阵形式。
