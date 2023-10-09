mindquantum.core.circuit.qfi
=============================

.. py:function:: mindquantum.core.circuit.qfi(circuit: Circuit, backend='mqvector')

    根据给定参数计算参数化量子线路的量子fisher信息。
    量子fisher信息定义如下：

    .. math::

        \text{QFI}_{i,j} = 4\text{Re}(A_{i,j} - B_{i,j})

    其中：

    .. math::

        A_{i,j} = \frac{\partial \left<\psi\right| }{\partial x_{i}}
        \frac{\partial \left|\psi\right> }{\partial x_{j}}

    并且：

    .. math::

        B_{i,j} = \frac{\partial \left<\psi\right| }{\partial x_i}\left|\psi\right>
        \left<\psi\right|\frac{\partial \left|\psi\right> }{\partial x_{j}}

    参数：
        - **circuit** (Circuit) - 一个给定的参数化量子线路。
        - **backend** (str) - 一个受支持的量子模拟器后端。请参考 :class:`~.simulator.Simulator` 的描述。默认值： ``'mqvector'``。

    返回：
        Function，一个函数，该函数输入参数化量子线路的值，返回量子fisher信息。
