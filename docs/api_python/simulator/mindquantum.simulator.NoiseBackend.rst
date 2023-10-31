mindquantum.simulator.NoiseBackend
==================================

.. py:class:: mindquantum.simulator.NoiseBackend(base_sim: str, n_qubits: int, adder: ChannelAdderBase, seed: int = None, dtype=None)

    基于噪声信道的含噪模拟器。

    参数：
        - **base_sim** (str) - MindQuantum 支持的量子模拟器。
        - **n_qubits** (int) - 该噪声模拟器的比特数。
        - **adder** (:class:`~.core.circuit.ChannelAdderBase`) - 一个信道添加器，可以将一个量子线路转化为噪声线路。
        - **seed** (int) - 一个随机种子。默认值： ``None``。
        - **dtype** (mindquantum.dtype) - 模拟器的数据类型。如果为 ``None``，将选取为 mindquantum.complex128。默认值： ``None``。
