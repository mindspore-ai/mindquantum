mindquantum.core.gates.XGate
===============================

.. py:class:: mindquantum.core.gates.XGate

    泡利X门。矩阵为：

    .. math::

        {\rm X}=\begin{pmatrix}0&1\\1&0\end{pmatrix}

    为了简单起见，我们将 ``X`` 定义为 ``XGate()`` 的实例。有关更多重新定义，请参考下面的功能表。

    .. note::
        为了简单起见，您可以在泡利门上执行指数运算（目前仅适用于泡利门）。规则如下：

        .. math::

            X^\theta = RX(\theta\pi)

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。
