.. py:function:: mindquantum.core.operators.count_qubits(operator)

    计算未使用的量子位被删除之前的量子位数目。

    .. note::
        在某些情况下，我们需要删除未使用的索引。

    **参数：**

    - **operator** (Union[FermionOperator, QubitOperator, QubitExcitationOperator]) - `operator` 算子类型为FermionOperator、QubitOperator或QubitExcitationOperator。

    **返回：**

    int，运算符作用的最小量子位数。

    **异常：**

    - **TypeError** - 类型无效的运算符。
