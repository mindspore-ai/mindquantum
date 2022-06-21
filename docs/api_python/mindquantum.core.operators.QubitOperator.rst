.. py:class:: mindquantum.core.operators.QubitOperator(term=None, coefficient=1.0)

    作用于量子位的项的总和，例如 0.5 * 'X1 X5' + 0.3 * 'Z1 Z2'。
    项是一个作用于n个量子位的运算符，可以表示为：coefficient * local_operator[0] x ... x local_operator[n-1]，其中x是张量乘积。
    本地运算符是作用于一个量子位的Pauli运算符（'I'，'X'，'Y'或者'Z'）。
    在数学符号中，一个QubitOperator是例如0.5 * 'X1 X5'的项，它意味着Pauli X运算符作用于量子位1和5，而恒等运算符作用于其余所有量子位。

    请注意，由QubitOperator算子组成的哈密顿量应该是一个厄米特算子，因此要求所有项的系数必须是实数。

    QubitOperator的属性设置如下：operators = ('X', 'Y', 'Z')，different_indices_commute = True。

    **参数：**

    - **term** (str) - 量子位运算符的输入项。默认值：None。
    - **coefficient** (Union[numbers.Number, str, ParameterResolver]) - 此量子位运算符的系数，可以是由字符串、符号或参数解析器表示的数字或变量。默认值：1.0。

    .. py:method:: count_gates()

        返回单哈密顿量处理时的门数量。

        **返回：**

        int，单量子门的数量。

    .. py:method:: dumps(indent=4)

        将QubitOperator转储到JSON（JavaScript对象表示法）。

        **参数：**

        - **indent** (int) - JSON数组元素和对象成员打印时的缩进。默认值：4。

        **返回：**

        JSON(strings)，QubitOperator的JSON字符串。

    .. py:method:: from_openfermion(of_ops)
        :staticmethod:

        将openfermion格式的玻色子运算符转换为mindquantum格式。

        **参数：**

        - **of_ops** (openfermion.QubitOperator) - openfermion框架中的玻色子算符。

        **返回：**

        QubitOperator，mindquantum框架中的玻色子算符。

    .. py:method:: imag
        :property:

        获得系数的虚部。

        **返回：**

        QubitOperator，此量子算符的虚部。

    .. py:method:: loads(strs)
        :staticmethod:

        将JSON（JavaScript对象表示法）加载到QubitOperator中。

        **参数：**

        - **strs** (str) - 转储的量子位运算符字符串。

        **返回：**

        FermionOperator，从字符串加载的QubitOperator。

    .. py:method:: matrix(n_qubits=None)

        将此量子位运算符转换为csr_matrix。

        **参数：**

        - **n_qubits** (int) - 结果矩阵的量子位数目。如果是None，则该值将是最大本地量子位数。默认值：None。

    .. py:method:: real
        :property:

        获得系数的实部。

        **返回：**

        QubitOperator，这个量子位运算符的实部。

    .. py:method:: split()

        将算符的系数跟算符本身分开。

        **返回：**

        List[List[ParameterResolver, QubitOperator]]，分裂后的结果。

    .. py:method:: to_openfermion()

        将量子位运算符转换为openfermion格式。
