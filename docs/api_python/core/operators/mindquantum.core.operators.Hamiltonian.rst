mindquantum.core.operators.Hamiltonian
=======================================

.. py:class:: mindquantum.core.operators.Hamiltonian(hamiltonian, dtype=None)

    QubitOperator哈密顿量的包装器。

    参数：
        - **hamiltonian** (Union[QubitOperator, scipy.sparse]) - 泡利量子比特算子或者一个稀疏矩阵。
        - **dtype** (mindquantum.dtype) - 哈密顿量的数据类型。默认值： ``None``。

    .. py:method:: astype(dtype)

        将哈密顿量转化为其他的数据类型。

        参数：
            - **dtype** (mindquantum.dtype) - 想要转化的数据类型。

    .. py:method:: dtype
        :property:

        获取哈密顿量的数据类型。

    .. py:method:: get_cpp_obj(hermitian=False)

        获得cpp对象。

        参数：
            - **hermitian** (bool) - 返回的cpp对象是否是原始哈密顿量的厄米共轭。

    .. py:method:: hamiltonian
        :property:

        获取哈密顿量的QubitOperator。

        异常：
            - **ValueError** - 如果哈密顿量是由稀疏矩阵构造的，则无法获取QubitOperator。

    .. py:method:: ham_termlist
        :property:

        获取哈密顿量的泡利项列表。

        异常：
            - **ValueError** - 如果哈密顿量是由稀疏矩阵构造的，则无法获取泡利项列表。

    .. py:method:: sparse(n_qubits=1)

        在后台计算哈密顿量的稀疏矩阵。

        参数：
            - **n_qubits** (int) - 哈密顿量的总量子比特数，仅在模式为'frontend'时需要。默认值： ``1``。

    .. py:method:: sparse_matrix
        :property:

        获取哈密顿量的稀疏矩阵表示。

        返回：
            scipy.sparse.csr_matrix，哈密顿量的稀疏矩阵表示。
