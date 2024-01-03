mindquantum.core.gates.Rxz
===============================

.. py:class:: mindquantum.core.gates.Rxz(pr)

    Rxz 门。更多用法，请参见 :class:`~.core.gates.RX`。

    .. math::

        {\rm Rxz_\theta}=\exp{\left(-i\frac{\theta}{2} Z\otimes X\right)} =\begin{pmatrix}
            \cos{\frac{\theta}{2}} & -i\sin{\frac{\theta}{2}} & 0 & 0\\
            -i\sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}} & 0 & 0\\
            0 & 0 & \cos{\frac{\theta}{2}} & i\sin{\frac{\theta}{2}}\\
            0 & 0 & i\sin{\frac{\theta}{2}} & \cos{\frac{\theta}{2}}\\
            \end{pmatrix}

    参数：
        - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: diff_matrix(pr=None, about_what=None)

        返回该参数化量子门的导数矩阵。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 该参数化量子门的参数值。默认值：None。
            - **about_what** (str) - 关于哪个参数求导数。输入值为str类型的对应参数名。默认值：None。

        返回：
            numpy.ndarray，该量子门的导数矩阵形式。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。

    .. py:method:: matrix(pr=None, full=False, **kwargs)

        返回该参数化量子门的矩阵。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 该参数化量子门的参数值。默认值：None。
            - **full** (bool) - 是否获取完整的矩阵（受控制比特和作用比特影响）。默认值： ``False``。
            - **kwargs** (dict) - 其他关键字参数。

        返回：
            numpy.ndarray，该量子门的矩阵形式。
