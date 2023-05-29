mindquantum.core.gates.HGate
=============================

.. py:class:: mindquantum.core.gates.HGate

    Hadamard门。矩阵为：

    .. math::

        {\rm H}=\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}

    更多用法，请参见 :class:`~.core.gates.XGate`。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。
