Class mindquantum.algorithm.nisq.Transform(operator, n_qubits=None)

    将费米子或者玻色子进行转化的模块。
    `jordan_wigner` , `parity` , `bravyi_kitaev` , `bravyi_kitaev_tree` , `bravyi_kitaev_superfast` 将会把 `FermionOperator` 转换为 `QubitOperator`。 `reversed_jordan_wigner` 将会把 `QubitOperator` 转换为 `FermionOperator` 。

    **参数：**

    - **operator** (Union[FermionOperator, QubitOperator]) - 需要进行转换的 `FermionOperator` 或 `QubitOperator` 。
    - **n_qubits** (int) - 输入算符的比特数。如果为 `None` ， 系统将会自动数出比特数。默认值：None。
