mindquantum.utils.random_hamiltonian
=====================================

.. py:function:: mindquantum.utils.random_hamiltonian(n_qubits, n_terms, seed=None, dtype=None)

    生成随机的泡利哈密顿量。

    参数：
        - **n_qubits** (int) - 量子比特数。
        - **n_terms** (int) - 泡利项的数量。
        - **seed** (int) - 随机种子。默认值： ``None``。
        - **dtype** (mindquantum.dtype) - 哈密顿量的数据类型。默认值： ``None``。

    返回：
        Hamiltonian，随机生成的哈密顿量。
