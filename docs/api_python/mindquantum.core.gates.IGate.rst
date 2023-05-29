mindquantum.core.gates.IGate
=============================

.. py:class:: mindquantum.core.gates.IGate

    Identity门。矩阵为：

    .. math::

        {\rm I}=\begin{pmatrix}1&0\\0&1\end{pmatrix}

    更多用法，请参见 :class:`~.core.gates.XGate`。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。
