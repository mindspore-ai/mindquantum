mindquantum.algorithm.nisq.StronglyEntangling
==============================================

.. py:class:: mindquantum.algorithm.nisq.StronglyEntangling(n_qubits: int, depth: int, entangle_gate: BasicGate, prefix: str = '', suffix: str = '')

    强纠缠ansatz。请参考 `Circuit-centric quantum classifiers <https://arxiv.org/pdf/1804.00633.pdf>`_。

    参数：
        - **n_qubits** (int) - ansatz作用于多少个量子比特。
        - **depth** (int) - ansatz的深度。
        - **entangle_gate** (BasicGate) - 产生纠缠的量子门。
          如果传入单量子比特门，则会添加一个控制量子比特，
          如果传入双量子比特门，则该双量子比特门将作用于不同的量子比特。
        - **prefix** (str) - 参数的前缀。默认值： ``''``。
        - **suffix** (str) - 参数的后缀。默认值： ``''``。
