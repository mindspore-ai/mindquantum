mindquantum.utils.random_circuit
=================================

.. py:function:: mindquantum.utils.random_circuit(n_qubits, gate_num, sd_rate=0.5, ctrl_rate=0.2, seed=None)

    生成随机线路。

    参数：
        - **n_qubits** (int) - 随机线路的量子比特数。
        - **gate_num** (int) - 随机线路中门的数量。
        - **sd_rate** (float) - 单量子门和双量子门的比例。
        - **ctrl_rate** (float) - 门具有控制位的可能性。
        - **seed** (int) - 生成随机线路的随机种子。
