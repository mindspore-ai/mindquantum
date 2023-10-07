mindquantum.core.circuit.QubitNumberConstrain
=============================================

.. py:class:: mindquantum.core.circuit.QubitNumberConstrain(n_qubits: int, with_ctrl: bool = True, add_after: bool = True)

    只将噪声信道作用在比特数为 ``n_qubits`` 的量子门上。

    参数：
        - **n_qubits** (int) - 量子门的比特数目。
        - **with_ctrl** (bool) - 控制比特是否算在总比特数目之内。默认值： ``True`` 。
        - **add_after** (bool) - 在量子门前面或者后面添加量子信道。默认值： ``True``。
