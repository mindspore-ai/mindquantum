mindquantum.core.gates.ZZ
===============================

.. py:class:: mindquantum.core.gates.ZZ(pr)

    伊辛ZZ门。更多用法，请参见 :class:`mindquantum.core.gates.RX`。

    .. note::
        `ZZ` 门已弃用，请使用 :class:`mindquantum.core.gates.Rzz`。

    .. math::

        {\rm ZZ_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_Z\otimes\sigma_Z

    参数：
        - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: diff_matrix(pr=None, about_what=None, frac=1)

        返回该参数化量子门的导数矩阵。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 该参数化量子门的参数值。默认值： ``None``。
            - **about_what** (str) - 关于哪个参数求导数。
            - **frac** (numbers.Number) - 系数的倍数。默认值： ``1``。

        返回：
            numpy.ndarray，该量子门的导数矩阵形式。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。

    .. py:method:: matrix(pr=None, frac=1)

        返回该参数化量子门的矩阵。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - 该参数化量子门的参数值。默认值： ``None``。
            - **frac** (numbers.Number) - 系数的倍数。默认值： ``1``。

        返回：
            numpy.ndarray，该量子门的矩阵形式。
