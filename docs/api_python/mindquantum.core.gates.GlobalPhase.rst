mindquantum.core.gates.GlobalPhase
===================================

.. py:class:: mindquantum.core.gates.GlobalPhase(pr)

    全局相位门。更多用法，请参见 :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm GlobalPhase}=\begin{pmatrix}\exp(-i\theta)&0\\
                        0&\exp(-i\theta)\end{pmatrix}

    参数：
        - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: diff_matrix(pr=None, about_what=None, **kwargs)

        参数门的倒数矩阵形式。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 参数门的参数。默认值： ``None``。
            - **about_what** (str) - 相对于哪个参数来求导数。默认值： ``None``。
            - **kwargs** (dict) - 其他参数。

        返回：
            numpy.ndarray，量子门的导数形式的矩阵。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。

    .. py:method:: matrix(pr=None, **kwargs)

        参数门的矩阵形式。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 参数门的参数。默认值： ``None``。
            - **kwargs** (dict) - 其他的参数。

        返回：
            numpy.ndarray，量子门的矩阵形式。
