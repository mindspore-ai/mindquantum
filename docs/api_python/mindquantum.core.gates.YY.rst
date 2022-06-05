mindquantum.core.gates.YYGate
=============================

.. py:class:: mindquantum.core.gates.YY(coeff=None)

    伊辛YY门。更多用法，请参见 :class:`mindquantum.core.gates.RX`。

    .. math::

        {\rm YY_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_y\otimes\sigma_y

    **参数：**

    - **coeff** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。默认值：None。
