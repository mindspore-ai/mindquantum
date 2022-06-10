.. py:class:: mindquantum.core.operators.FermionOperator(term=None, coefficient=1.0)

    费米子算子，如FermionOperator('4^ 3 9 3^')表示 :math:`a_4^\dagger a_3 a_9 a_3^\dagger`。
    这些是描述费米子系统的基本运算符，如分子系统。
    FermionOperator遵循反交换关系。

    **参数：**

    - **terms** (str) - 费米子算子的输入项。默认值：None。
    - **coefficient** (Union[numbers.Number, str, ParameterResolver]) - 对应单运算符的系数。默认值：1.0。

    .. py:method:: dumps(indent=4)

        将FermionOperator转储到JSON（JavaScript对象表示法）。

        **参数：**

        - **indent** (int) - JSON数组元素和对象成员打印时的缩进大小。默认值：4。

        **返回：**

        JSON(str)，FermionOperator的JSON字符串。

    .. py:method:: from_openfermion(of_ops)
        :staticmethod:

        将openfermion格式的费米子运算符转换为mindquantum格式。

        **参数：**

        - **of_ops** (openfermion.FermionOperator) - openfermion中的费米子算符。

        **返回：**

        FermionOperator，mindquantum中的费米子算符。

    .. py:method:: imag
        :property:

        获得系数的虚部。

        **返回：**

        FermionOperator，这个FermionOperator的虚部。

    .. py:method:: loads(strs)
        :staticmethod:

        将JSON（JavaScript对象表示法）加载到FermionOperator中。

        **参数：**

        - **strs** (str) - 转储的费米子运算符字符串。

        **返回：**

        FermionOperator，从字符串加载的FermionOperator。

    .. py:method:: matrix(n_qubits=None)

        将此费米子运算符转换为jordan_wigner映射下的csr_matrix。

        **参数：**

        - **n_qubits** (int) - 结果矩阵的总量子位。如果是None，则该值将是最大本地量子位数。默认值：None。

    .. py:method:: normal_ordered()

        返回FermionOperator的规范有序形式。

        **返回：**

        FermionOperator，规范有序的FermionOperator。

    .. py:method:: real
        :property:

        获得系数的实部。

        **返回：**

        FermionOperator，这个FermionOperator的实部。

    .. py:method:: split()

        将算符的系数跟算符本身分开。

        **返回：**

        List[List[ParameterResolver, FermionOperator]]，分裂后的结果。

    .. py:method:: to_openfermion()

        将费米子运算符转换为openfermion格式。
