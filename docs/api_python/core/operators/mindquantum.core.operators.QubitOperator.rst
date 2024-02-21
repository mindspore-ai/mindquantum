mindquantum.core.operators.QubitOperator
=========================================

.. py:class:: mindquantum.core.operators.QubitOperator(terms: typing.Union[str, "QubitOperator"] = None, coefficient: PRConvertible = 1.0, internal: bool = False)

    作用于量子比特的项的总和，例如 0.5 * 'X1 X5' + 0.3 * 'Z1 Z2'。
    项是一个作用于n个量子比特的运算符，可以表示为：coefficient * local_operator[0] x ... x local_operator[n-1]，其中x是张量乘积。
    本地运算符是作用于一个量子比特的泡利运算符（'I'，'X'，'Y'或者'Z'）。
    在数学符号中，一个QubitOperator是例如0.5 * 'X1 X5'的项，它意味着X运算符作用于量子比特1和5，而恒等运算符作用于其余所有量子比特。

    请注意，由QubitOperator算子组成的哈密顿量应该是一个厄米算子，因此要求所有项的系数必须是实数。

    QubitOperator的属性设置如下：operators = ('X', 'Y', 'Z')，different_indices_commute = True。

    参数：
        - **term** (Union[str, QubitOperator]) - 量子比特运算符的输入项。默认值： ``None``。
        - **coefficient** (Union[numbers.Number, str, Dict[str, numbers.Number], ParameterResolver]) - 此量子比特运算符的系数，可以是由字符串、符号或参数解析器表示的数字或变量。默认值： ``1.0``。
        - **internal** (bool) - 第一个参数是否为泡利算符对象的内部c++类。默认值： ``False``。

    .. py:method:: astype(dtype)

        将QubitOperator转化为不同的数据类型。

        .. note::
            将一个复数类型的QubitOperator转化为实数类型将会忽略系数的虚数部分。

        参数：
            - **dtype** (mindquantum.dtype) - 玻色子算符的新类型。

        返回：
            QubitOperator，给定类型的玻色子算符。

    .. py:method:: cast_complex()

        将一个玻色子算符转化为等价的复数形式。

    .. py:method:: compress(abs_tol=EQ_TOLERANCE)

        将系数很小的玻色子串项移除。

        参数：
          - **abs_tol** (float) - 绝对值阈值，必须大于0.0。默认值： ``EQ_TOLERANCE``。

        返回：
            QubitOperator，经过压缩后的玻色子算符。

    .. py:method:: count_gates()

        返回单哈密顿量处理时的门数量。

        返回：
            int，单量子门的数量。

    .. py:method:: count_qubits()

        统计移除没用比特前所占用的比特数。

        返回：
            int，移除没用比特前所占用的比特数。

    .. py:method:: dtype
        :property:

        玻色子算符系数的数据类型。

    .. py:method:: dumps(indent: int = 4)

        将QubitOperator转储到JSON（JavaScript对象表示法）。

        参数：
            - **indent** (int) - JSON数组元素和对象成员打印时的缩进。默认值： ``4``。

        返回：
            JSON(strings)，QubitOperator的JSON字符串。

    .. py:method:: from_openfermion(of_ops)
        :staticmethod:

        将openfermion格式的玻色子算符转换为mindquantum格式。

        参数：
            - **of_ops** (openfermion.QubitOperator) - openfermion框架中的玻色子算符。

        返回：
            QubitOperator，mindquantum框架中的玻色子算符。

    .. py:method:: get_coeff(term)

        获取给定项的系数。

        参数：
            - **term** (List[Tuple[int, Union[int, str]]]) - 想要获取系数的项。

    .. py:method:: hermitian()

        返回QubitOperator的厄米共轭。

        返回：
            QubitOperator，玻色子算符的厄米共轭。

    .. py:method:: imag
        :property:

        获得系数的虚部。

        返回：
            QubitOperator，此玻色子算符的虚部。

    .. py:method:: is_complex
        :property:

        返回当前玻色子是否使用复数类型的系数。

    .. py:method:: is_singlet
        :property:

        检查当前玻色子是否只有一项。

        返回：
            bool，当前玻色子是否只有一项。

    .. py:method:: loads(strs: str)
        :staticmethod:

        将JSON（JavaScript对象表示法）加载到QubitOperator中。

        参数：
            - **strs** (str) - 转储的玻色子算符字符串。

        返回：
            QubitOperator，从字符串加载的QubitOperator。

    .. py:method:: matrix(n_qubits: int = None, pr=None)

        将此玻色子算符转换为csr_matrix。

        参数：
            - **n_qubits** (int) - 结果矩阵的量子比特数目。如果是None，则该值将是最大局域量子比特数。默认值： ``None``。
            - **pr** (ParameterResolver, dict, numpy.ndarray, list, numbers.Number) - 含参玻色子算符的参数。默认值： ``None``。

    .. py:method:: parameterized
        :property:

        检查当前玻色子是否是参数化的。

    .. py:method:: params_name
        :property:

        获取玻色子算符的所有参数。

    .. py:method:: real
        :property:

        获得系数的实部。

        返回：
            QubitOperator，这个玻色子算符的实部。

    .. py:method:: relabel(logic_qubits: typing.List[int])

        根据逻辑比特顺序重新编码量子比特。

        参数：
            - **logic_qubits** (List[int]) - 逻辑比特编号。

    .. py:method:: singlet()

        将只有一个费米子串的玻色子算符分裂成只有一个玻色子的玻色子算符。

        返回：
            List[QubitOperator]，只有一个玻色子的玻色子算符。

        异常：
            - **RuntimeError** - 如果该玻色子算符拥有不止一个玻色子串。

    .. py:method:: singlet_coeff()

        当玻色子算符只有一个玻色子串时，返回该玻色子串的系数。

        返回：
            ParameterResolver，唯一玻色子串的系数。

        异常：
            - **RuntimeError** - 如果该玻色子算符拥有不止一个玻色子串。

    .. py:method:: size
        :property:

        返回玻色子算符中玻色子串的数量。

    .. py:method:: split()

        将算符的系数跟算符本身分开。

        返回：
            List[List[ParameterResolver, QubitOperator]]，分裂后的结果。

    .. py:method:: subs(params_value: PRConvertible)

        将玻色子中的变量换成具体的参数值。

        参数：
            - **params_value** (Union[Dict[str, numbers.Number], ParameterResolver]) - 系数变量的参数。

    .. py:method:: terms
        :property:

        返回玻色子算符中的玻色子串。

    .. py:method:: to_openfermion()

        将玻色子算符转换为openfermion格式。
