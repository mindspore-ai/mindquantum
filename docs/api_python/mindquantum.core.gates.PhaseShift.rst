.. py:class:: mindquantum.core.gates.PhaseShift(pr)

    相移门。更多用法，请参见 :class:`mindquantum.core.gates.RX`。

    .. math::

        {\rm PhaseShift}=\begin{pmatrix}1&0\\
                         0&\exp(i\theta)\end{pmatrix}

    **参数：**

    - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: matrix(pr=None)

        参数门的矩阵形式。

        **参数：**

        - **pr** (Union[ParameterResolver, dict]) - 参数门的矩阵形式。默认值：None。

        **返回：**

        numpy.ndarray，量子门的矩阵形式。

    .. py:method:: diff_matrix(pr=None, about_what=None)

        参数门的倒数矩阵形式。

        **参数：**

        - **pr** (Union[ParameterResolver, dict]) - 量子门的参数。默认值：None。
        - **about_what** (str) - 相对于哪个参数来求导数。默认值：None。

        **返回：**

        numpy.ndarray, 量子门的导数形式的矩阵。
