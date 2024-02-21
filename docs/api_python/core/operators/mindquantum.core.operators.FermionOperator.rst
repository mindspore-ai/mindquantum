mindquantum.core.operators.FermionOperator
===========================================

.. py:class:: mindquantum.core.operators.FermionOperator(terms: typing.Union[str, "FermionOperator"] = None, coefficient: PRConvertible = 1.0, internal: bool = False)

    费米子算子，如FermionOperator('9 4^ 3 3^')表示 :math:`a_9 a_4^\dagger a_3 a_3^\dagger`。
    这些是描述费米子系统的基本运算符，如分子系统。
    FermionOperator遵循反交换关系。

    参数：
        - **terms** (Union[str, ParameterResolver]) - 费米子算符的输入项。默认值： ``None``。
        - **coefficient** (Union[numbers.Number, str, Dict[str, numbers.Number], ParameterResolver]) - 单费米子算符的系数。默认值： ``1.0``。
        - **internal** (bool) - 第一个参数是否为费米子对象的内部c++类。默认值： ``False``。

    .. py:method:: astype(dtype)

        将FermionOperator转化为不同的数据类型。

        .. note::
            将一个复数类型的FermionOperator转化为实数类型将会忽略系数的虚数部分。

        参数：
            - **dtype** (mindquantum.dtype) - 费米子算符的新类型。

        返回：
            FermionOperator，给定类型的费米子算符。

    .. py:method:: cast_complex()

        将一个费米子算符转化为等价的复数形式。

    .. py:method:: compress(abs_tol=EQ_TOLERANCE)

        将系数很小的费米子串项移除。

        参数：
          - **abs_tol** (float) - 绝对值阈值，必须大于0.0。默认值： ``EQ_TOLERANCE``。

        返回：
            FermionOperator，经过压缩后的费米子算符。

    .. py:method:: constant
        :property:

        返回单位费米子串的系数。

        返回：
            ParameterResolver，单位费米子串的系数。

    .. py:method:: count_qubits()

        统计移除没用比特前所占用的比特数。

        返回：
            int，移除没用比特前所占用的比特数。

    .. py:method:: dtype
        :property:

        费米子算符系数的数据类型。

    .. py:method:: dumps(indent: int = 4)

        将FermionOperator转储到JSON（JavaScript对象表示法）。

        参数：
            - **indent** (int) - JSON数组元素和对象成员打印时的缩进大小。默认值： ``4``。

        返回：
            JSON(str)，FermionOperator的JSON字符串。

    .. py:method:: from_openfermion(of_ops)
        :staticmethod:

        将openfermion格式的费米子运算符转换为mindquantum格式。

        参数：
            - **of_ops** (openfermion.FermionOperator) - openfermion中的费米子算符。

        返回：
            FermionOperator，mindquantum中的费米子算符。

    .. py:method:: get_coeff(term)

        获取给定项的系数。

        参数：
            - **term** (List[Tuple[int, Union[int, str]]]) - 想要获取系数的项。

    .. py:method:: hermitian()

        返回费米子运算符的厄米共轭。

        返回：
            FermionOperator，原费米子的厄米共轭。

    .. py:method:: imag
        :property:

        获得系数的虚部。

        返回：
            FermionOperator，这个FermionOperator的虚部。

    .. py:method:: is_complex
        :property:

        返回当前费米子是否使用复数类型的系数。

    .. py:method:: is_singlet
        :property:

        检查当前费米子是否只有一项。

        返回：
            bool，当前费米子是否只有一项。

    .. py:method:: loads(strs: str)
        :staticmethod:

        将JSON（JavaScript对象表示法）加载到FermionOperator中。

        参数：
            - **strs** (str) - 转储的费米子运算符字符串。

        返回：
            FermionOperator，从字符串加载的FermionOperator。

    .. py:method:: matrix(n_qubits: int = None, pr=None)

        将此费米子运算符转换为jordan_wigner映射下的csr_matrix。

        参数：
            - **n_qubits** (int) - 结果矩阵的总量子比特数。如果是None，则该值将是最大局域量子比特数。默认值： ``None``。
            - **pr** (ParameterResolver, dict, numpy.ndarray, list, numbers.Number) - 含参费米子算符的参数。默认值： ``None``。

    .. py:method:: normal_ordered()

        返回FermionOperator的规范有序形式。

        返回：
            FermionOperator，规范有序的FermionOperator。

    .. py:method:: parameterized
        :property:

        检查当前费米子是否是参数化的。

    .. py:method:: params_name
        :property:

        获取费米子算符的所有参数。

    .. py:method:: real
        :property:

        获得系数的实部。

        返回：
            FermionOperator，这个FermionOperator的实部。

    .. py:method:: relabel(logic_qubits: typing.List[int])

        根据逻辑比特顺序重新编码量子比特。

        参数：
            - **logic_qubits** (List[int]) - 逻辑比特编号。

    .. py:method:: singlet()

        将只有一个费米子串的费米子算符分裂成只有一个费米子的费米子算符。

        返回：
            List[FermionOperator]，只有一个费米子的费米子算符。

        异常：
            - **RuntimeError** - 如果该费米子算符拥有不止一个费米子串。

    .. py:method:: singlet_coeff()

        当费米子算符只有一个费米子串时，返回该费米子串的系数。

        返回：
            ParameterResolver，唯一费米子串的系数。

        异常：
            - **RuntimeError** - 如果该费米子算符拥有不止一个费米子串。

    .. py:method:: size
        :property:

        返回费米子算符中费米子串的数量。

    .. py:method:: split()

        将算符的系数跟算符本身分开。

        返回：
            List[List[ParameterResolver, FermionOperator]]，分裂后的结果。

    .. py:method:: subs(params_value: PRConvertible)

        将费米子中的变量换成具体的参数值。

        参数：
            - **params_value** (Union[Dict[str, numbers.Number], ParameterResolver]) - 系数变量的参数。

    .. py:method:: terms
        :property:

        返回费米子算符中的费米子串。

    .. py:method:: to_openfermion()

        将费米子运算符转换为openfermion格式。
