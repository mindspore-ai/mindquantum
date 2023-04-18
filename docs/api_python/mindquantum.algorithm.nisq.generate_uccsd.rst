mindquantum.algorithm.nisq.generate_uccsd
==========================================

.. py:function:: mindquantum.algorithm.nisq.generate_uccsd(molecular, threshold=0)

    使用OpenFermion生成的分子数据生成uccsd量子线路。

    参数：
        - **molecular** (Union[str, MolecularData]) - 分子数据文件的名称，或openfermion中的 `MolecularData` 。
        - **threshold** (float) - 过滤uccsd中组态幅度的阈值。我们将保留那些组态振幅绝对值比 `threshold` 大的组态，因此，当 `threshold=0` 时，只会保留非零振幅的组态。默认值： ``0``。

    返回：
        - **uccsd_circuit** (Circuit) - 由uccsd方法生成的ansatz电路。
        - **initial_amplitudes** (numpy.ndarray) - uccsd电路的初始参数值。
        - **parameters_name** (list[str]) - 初始参数的名称。
        - **qubit_hamiltonian** (QubitOperator) - 分子的哈密顿量。
        - **n_qubits** (int) - 模拟中使用的量子比特数。
        - **n_electrons** (int) - 分子的电子数。
