mindquantum.core.operators.count_qubits(operator)

    计算运算符之前作用的量子比特数删除未使用的量子位

    注:
        在某些情况下，我们需要删除未使用的索引。

    参数:
        operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]):
            费米子操作符或Qubit操作符或Qubit激励操作符。

    返回:
        int，运算符操作的最小量子位数。

    异常:
       TypeError: 类型无效的运算符。

    样例:
        >>> from mindquantum.core.operators import QubitOperator,FermionOperator
        >>> from mindquantum.core.operators.utils import count_qubits
        >>> qubit_op = QubitOperator("X1 Y2")
        >>> count_qubits(qubit_op)
        3
        >>> fer_op = FermionOperator("1^")
        >>> count_qubits(fer_op)
        2
    