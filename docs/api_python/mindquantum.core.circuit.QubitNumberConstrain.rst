mindquantum.core.circuit.QubitNumberConstrain
=============================================

.. py:class:: mindquantum.core.circuit.QubitNumberConstrain(adder: ChannelAdderBase)

    翻转给定信道添加器的接受和拒绝规则。

    参数：
        - **n_qubits** (int) - 量子门的比特数目。
        - **with_ctrl** (bool) - 控制比特是否算在总比特数目之内。默认值： ``True`` 。
        - **add_after** (bool) - 在量子门前面或者后面添加量子信道。默认值： ``True``。
