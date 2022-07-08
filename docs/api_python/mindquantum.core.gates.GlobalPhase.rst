.. py:class:: mindquantum.core.gates.GlobalPhase(pr)

    全局相位门。更多用法，请参见 :class:`mindquantum.core.gates.RX`.

    .. math::

        {\rm GlobalPhase}=\begin{pmatrix}\exp(-i\theta)&0\\
                        0&\exp(-i\theta)\end{pmatrix}

    **参数：**

    - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。
