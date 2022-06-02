.. py:method:: mindquantum.simulator.Simulator.get_expectation(hamiltonian)

    得到给定hamiltonian的期望。hamiltonian可能是非厄密共轭的。

    .. math::

        E = \left<\psi\right|H\left|\psi\right>

    **参数：**

    - **hamiltonian** (Hamiltonian) - 想得到期望的hamiltonian。

    **返回：**

    numbers.Number，期望值。        