.. py:function:: mindquantum.core.circuit.decompose_single_term_time_evolution(term, para)

    将时间演化门分解成基本的量子门。

    这个函数只适用于只有单个Pauli词的hamiltonian。
    例如，exp(-i * t * ham), ham只能是一个Pauli词，如ham = X0 x Y1 x Z2, 此时，结果是((0, 'X'), (1, 'Y'), (2, 'Z'))。
    当演化时间被表示成t = a*x + b*y时，参数将是{'x':a, 'y':b}。

    **参数：**

    - **term** (tuple, QubitOperator) - 仅演化量子算子的hamiltonian项。
    - **para** (Union[dict, numbers.Number]) - 演化算子的参数。

    **返回：**

    Circuit，量子线路。

    **异常：**

    - **ValueError** - 如果term里有多个pauli字符串。
    - **TypeError** - 如果term不是map。
