mindquantum.algorithm.nisq.uccsd_singlet_generator
===================================================

.. py:function:: mindquantum.algorithm.nisq.uccsd_singlet_generator(n_qubits, n_electrons, anti_hermitian=True)

    为 `n_electrons` 的系统生成单态UCCSD算子。此函数生成一个由费米子构成的UCCSD算子，该算子作用在一个由 `n_qubits` 的自旋轨道和 `n_electrons` 电子构成的单参考态，也就是自旋单态算符，这也意味着该算符能够保证自旋守恒。

    参数：
        - **n_qubits** (int) - 用于表示系统的自旋轨道数，这也对应于非紧凑映射中的量子比特数。
        - **n_electrons** (int) - 物理系统中电子数。
        - **anti_hermitian** (bool) - 仅生成普通CCSD运算符而不是幺正的形式，主要用于测试。

    返回：
        FermionOperator，构建UCCSD波函数的UCCSD算子。
