mindquantum.core.gates.GroupedPauli
===================================

.. py:class:: mindquantum.core.gates.GroupedPauli(pauli_string: str)

    多比特泡利串旋转门。

    组合的泡利串门可以将泡利算符同时作用到量子态上，这将会加速量子线路的演化。

    .. math::

        U =\otimes_i\sigma_i, \text{where } \sigma \in \{I, X, Y, Z\}

    参数：
        - **pauli_string** (str) - 泡利串。泡利串中的元素只能是 ``['i', 'x', 'y', 'z', 'I', 'X', 'Y', 'Z']``。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。

    .. py:method:: matrix(full=False)

        返回该量子门的矩阵。

        参数：
            - **full** (bool) - 是否获取完整的矩阵（受控制比特和作用比特影响）。默认值： ``False``。

        返回：
            numpy.ndarray，该量子门的矩阵形式。
