mindquantum.core.gates.SWAPalpha
=================================

.. py:class:: mindquantum.core.gates.SWAPalpha(pr)

    SWAPalpha 门。更多用法，请参见 :class:`~.core.gates.RX`。

    .. math::

        \text{SWAP}(\alpha) =
        \begin{pmatrix}
            1 & 0 & 0 & 0\\
            0 & \frac{1}{2}\left(1+e^{i\pi\alpha}\right) & \frac{1}{2}\left(1-e^{i\pi\alpha}\right) & 0\\
            0 & \frac{1}{2}\left(1-e^{i\pi\alpha}\right) & \frac{1}{2}\left(1+e^{i\pi\alpha}\right) & 0\\
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
