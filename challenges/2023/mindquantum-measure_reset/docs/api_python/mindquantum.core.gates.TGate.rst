mindquantum.core.gates.TGate
===============================

.. py:class:: mindquantum.core.gates.TGate

    T门。矩阵为：

    .. math::
        {\rm T}=\begin{pmatrix}1&0\\0&(1+i)/\sqrt(2)\end{pmatrix}

    更多用法，请参见 :class:`~.core.gates.XGate`。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。
