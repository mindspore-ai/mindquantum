.. py:class:: mindquantum.core.operators.Hamiltonian(hamiltonian)

    QubitOperator哈密顿量的包装器。

    **参数：**

    - **hamiltonian** (QubitOperator) - 泡利量子比特算子。

    .. py:method:: get_cpp_obj(hermitian=False)

        获得cpp对象。

        **参数：**

        - **hermitian** (bool) - 是否获得哈密顿量的cpp对象的厄密共轭。

    .. py:method:: sparse(n_qubits=1)

        在pqc算子中计算哈密顿量的稀疏矩阵。

        **参数：**

        - **n_qubits** (int) - 哈密顿量的总量子比特数，仅在模式为'frontend'时需要。默认值：1。
