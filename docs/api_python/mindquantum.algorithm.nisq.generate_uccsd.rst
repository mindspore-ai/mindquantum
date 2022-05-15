mindquantum.algorithm.nisq.generate_uccsd(molecular, th=0)

    使用HiQ费米离子或开放费米离子根据生成的分子数据生成uccsd量子电路。

    参数:
        molecular (Union[str, MolecularData]): 分子数据文件的名称，或openfermion分子数据。
        th (int): 过滤uccsd幅度的阈值。当th < 0时，我们将保留所有振幅。当th == 0时，我们将保留所有正振幅。默认值：0。

    返回:
        - **uccsd_circuit** (Circuit), 由uccsd方法生成的asatz电路。
        - **initial_amplitudes** (numpy.ndarray), uccsd电路的初始参数值。
        - **parameters_name** (list[str]), 初始参数的名称。
        - **qubit_hamiltonian** (QubitOperator), 分子的汉密尔顿。
        - **n_qubits** (int), 模拟中的量子位数。
        - **n_electrons**, 分子的电子数。
    