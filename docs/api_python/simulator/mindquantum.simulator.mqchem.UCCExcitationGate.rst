mindquantum.simulator.mqchem.UCCExcitationGate
===============================================

.. py:class:: mindquantum.simulator.mqchem.UCCExcitationGate(fermion_operator)

    酉耦合簇（UCC）激发门，用于与 :class:`~.simulator.mqchem.MQChemSimulator` 一同使用。

    该门表示酉算符 :math:`e^{G}`，其中 :math:`G` 是一个反厄米生成元，形式为 :math:`G = \theta (T - T^\dagger)`。
    这里，:math:`T` 是一个同时保持自旋和电子数守恒的费米子激发算符。该算符常用于量子化学中变分量子算法的UCC拟设。

    .. note::
        此门专为 `MQChemSimulator` 设计，并依赖其内部的CI空间表示。它与标准的态矢量 `Simulator` 不兼容。

    该门定义为：

    .. math::

        U(\theta) = \exp(\theta(T - T^\dagger))

    其中 :math:`T` 必须是单项的 :class:`~.core.operators.FermionOperator`，
    例如 :math:`a_p^\dagger a_q`。

    参数：
        - **fermion_operator** (FermionOperator) - 费米子激发算符 :math:`T`。
          它必须只包含一项。该项的系数用作旋转角 :math:`\theta`。如果系数是变量，则该门是参数化的。
