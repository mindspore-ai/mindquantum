.. py:class:: mindquantum.core.gates.XX(pr)

    伊辛XX门。更多用法，请参见 :class:`mindquantum.core.gates.RX`。

    .. math::

        {\rm XX_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_x\otimes\sigma_x

    **参数：**

    - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。
