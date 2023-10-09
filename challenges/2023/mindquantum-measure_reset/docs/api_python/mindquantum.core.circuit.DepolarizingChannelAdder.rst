mindquantum.core.circuit.DepolarizingChannelAdder
=================================================

.. py:class:: mindquantum.core.circuit.DepolarizingChannelAdder(p: float, n_qubits: int, add_after: bool = True)

    去极化信道添加器。

    参数：
        - **p** (float) - 去极化信道概率。
        - **n_qubits** (int) - 去极化信道的比特数。
        - **add_after** (bool) - 在量子门前面或者后面添加量子信道。默认值： ``True``。
