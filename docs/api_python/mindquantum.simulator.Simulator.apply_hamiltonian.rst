.. py:method:: mindquantum.simulator.Simulator.apply_hamiltonian(hamiltonian: mindquantum.core.operators.hamiltonian.Hamiltonian)

    将hamiltonian应用到模拟器上，这个hamiltonian可以是hermitian或non hermitian。

    .. note::
        应用hamiltonian后，量子态可能不是归一化量子态。

    **参数：**

    - **hamiltonian** (Hamiltonian) - 想应用的hamiltonian。