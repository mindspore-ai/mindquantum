.. py:class:: mindquantum.core.gates.PhaseShift(pr)

    相移门。更多用法，请参见 :class:`mindquantum.core.gates.RX`。

    .. math::

        {\rm PhaseShift}=\begin{pmatrix}1&0\\
                         0&\exp(i\theta)\end{pmatrix}

    **参数：**

    - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: matrix(pr=None)

        获取此参数化门的矩阵。

        **参数：**

        - **pr** (Union[ParameterResolver, dict]) - 参数门的参数值。默认值： `None` 。

    .. py:method:: diff_matrix(pr=None, about_what=None)

        获取此参数化门的导数矩阵。

        **参数：**

        - **pr** (Union[ParameterResolver, dict]) - 参数门的参数值。默认值： `None` 。
        - **about_what** (str) - 相对于哪个参数求导数。默认值： `None` 。
