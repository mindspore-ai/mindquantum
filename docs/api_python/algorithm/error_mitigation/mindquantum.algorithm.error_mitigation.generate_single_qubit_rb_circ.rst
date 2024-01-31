mindquantum.algorithm.error_mitigation.generate_single_qubit_rb_circ
=====================================================================

.. py:function:: mindquantum.algorithm.error_mitigation.generate_single_qubit_rb_circ(length: int, seed: int = None)

    生成单比特量子随机基准测试线路。

    参数：
        - **length** (int) - 线路中clifford元的个数。
        - **seed** (int) - 用于生成随机基准测试线路的随机数种子。如果为 ``None``，将会使用一个随机的种子。默认值： ``None``。

    返回：
        :class:`~.core.circuit.Circuit`，单比特随机基准测试线路，线路的模态为零态。
