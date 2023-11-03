mindquantum.core.parameterresolver.ParameterResolver
====================================================

.. py:class:: mindquantum.core.parameterresolver.ParameterResolver(data=None, const=None, dtype=None, internal=False)

    ParameterResolver可以设置参数化量子门或参数化量子线路的参数。

    参数：
        - **data** (Union[dict, numbers.Number, str, ParameterResolver]) - 初始参数名称及其值。如果数据是dict，则键将是参数名称，值将是参数值。如果数据是数字，则此数字将是此参数解析器的常量值。如果数据是字符串，则此字符串将是系数为1的唯一参数。默认值： ``None``。
        - **const** (number.Number) - 此参数解析器的常量部分。默认值： ``None``。
        - **dtype** (mindquantum.dtype) - 改参数解析器的数据类型。默认： ``None``。
        - **internal** (bool) - 第一个参数是否时参数解析器的c++对象。默认： ``False``。

    .. py:method:: ansatz_parameters
        :property:

        获取参数，该参数是一个ansatz参数。

        返回：
            set，ansatz参数的集合。

    .. py:method:: ansatz_part(*names)

        设置哪个部分是ansatz参数。

        参数：
            - **names** (tuple[str]) - 作为一个ansatz参数。

        返回：
            ParameterResolver，参数解析器本身。


    .. py:method:: as_ansatz()

        将所有参数设置为ansatz。

        返回：
            ParameterResolver，参数解析器。

    .. py:method:: as_encoder()

        将所有参数设置为编码器。

        返回：
            ParameterResolver，参数解析器。

    .. py:method:: astype(dtype)

        将参数解析器转变为其他数据类型。

        参数：
            - **dtype** (mindquantum.dtype) - 参数解析器的新的数据类型。

    .. py:method:: combination(other: typing.Union[typing.Dict[str, numbers.Number], "ParameterResolver"])

        将该参数解析器与输入的参数解析器进行线性组合。

        参数：
            - **other** (Union[Dict[str, numbers.Number], ParameterResolver]) - 需要做线性组合的参数解析器。

        返回：
            numbers.Number，组合结果。

    .. py:method:: conjugate()

        获取参数解析器的共轭。

        返回：
            ParameterResolver，参数解析器的共轭版本。

    .. py:method:: const
        :property:

        获取此参数解析器的常量部分。

        返回：
            numbers.Number，此参数解析器的常量部分。

    .. py:method:: dtype
        :property:

        获取参数解析器的数据类型。

    .. py:method:: dumps(indent=4)

        将参数解析器转储到JSON（JavaScript对象表示法）。

        .. note::
            由于float32类型的数据不能够序列化，因此 ``mindquantum.float32`` 和 ``mindquantum.complex64`` 类型的参数解析器也不能够被序列化。

        参数：
            - **indent** (int) - 打印JSON数据时的缩进级别，利用缩进会使打印效果更加美观。默认值： ``4``。

        返回：
            string(JSON)，参数解析器的JSON。

    .. py:method:: encoder_parameters
        :property:

        获取所有encoder参数。

        返回：
            set，encoder参数构成的集合。

    .. py:method:: encoder_part(*names)

        设置哪一部分是编码器参数。

        参数：
            - **names** (tuple[str]) - 用作编码器的参数。

        返回：
            ParameterResolver，参数解析器本身。

    .. py:method:: expression()

        获取此参数解析器的表达式字符串。

        返回：
            str，此参数解析器的字符串表达式。

    .. py:method:: imag
        :property:

        获取每个参数值的虚部构成的参数解析器。

        返回：
            ParameterResolver，参数解析器的虚部。

    .. py:method:: is_anti_hermitian()

        检查该参数解析器的参数值是否为反厄米。

        返回：
            bool，参数解析器是否为反厄米。

    .. py:method:: is_complex()
        :property:

        返回此参数解析器实例当前是否正在使用复数系数。

    .. py:method:: is_const()

        检查此参数解析器是否表示常量，这意味着此参数解析器中没有具有非零系数的参数。

        返回：
            bool，此参数解析器是否表示常量。

    .. py:method:: is_hermitian()

        检查该参数解析器的参数值是否为厄米的。

        返回：
            bool，参数解析器是否为厄米的。

    .. py:method:: items()

        生成所有参数的名称和值的迭代器。

    .. py:method:: keys()

        生成所有参数名称的迭代器。

    .. py:method:: loads(strs: str)
        :staticmethod:

        将JSON（JavaScript对象表示法）加载到FermionOperator中。

        参数：
            - **strs** (str) - 转储参数解析器字符串。

        返回：
            FermionOperator，从字符串加载的FermionOperator。

    .. py:method:: no_grad()

        将所有参数设置为不需要计算梯度。该操作为原地操作。

        返回：
            ParameterResolver，参数解析器本身。

    .. py:method:: no_grad_parameters
        :property:

        获取不需要计算梯度的参数。

        返回：
            set，不需要计算梯度的参数集合。

    .. py:method:: no_grad_part(*names)

        设置不需要梯度的部分参数。

        参数：
            - **names** (tuple[str]) - 不需要计算梯度的参数。

        返回：
            ParameterResolver，参数解析器本身。

    .. py:method:: params_name
        :property:

        获取参数名称。

        返回：
            list，参数名称的列表。

    .. py:method:: params_value
        :property:

        获取参数值。

        返回：
            list，参数值的列表。

    .. py:method:: pop(v: str)

        弹出参数。

        参数：
            - **v** (str) - 想要弹出的参数名称。

        返回：
            numbers.Number，弹出的参数值。

    .. py:method:: real
        :property:

        获取每个参数值的实部。

        返回：
            ParameterResolver，参数值的实部。


    .. py:method:: requires_grad()

        将此参数解析器的所有参数设置为需要进行梯度计算。该操作为原地操作。

        返回：
            ParameterResolver，参数解析器本身。
    .. py:method:: requires_grad_parameters
        :property:

        获取需要梯度的参数。

        返回：
            set，需要计算梯度的参数集合。

    .. py:method:: requires_grad_part(*names)

        设置部分需要计算梯度的参数。该操作为原地操作。

        参数：
            - **names** (tuple[str]) - 需要梯度的参数。

        返回：
            ParameterResolver，参数解析器本身。

    .. py:method:: subs(other: typing.Union["ParameterResolver", typing.Dict[str, numbers.Number]])

        将变量的参数值带入参数解析器。

        参数：
            - **other** (Union[ParameterResolver, Dict[str, numbers.Number]]) - 参数解析器中的变量的值。

    .. py:method:: to_real_obj()

        转化为实数类型。

    .. py:method:: update(other: "ParameterResolver")

        使用其它参数解析器更新此参数解析器。

        参数：
            - **other** (ParameterResolver) - 其它参数解析器。

        异常：
            - **ValueError** - 如果某些参数需要grad而在其它参数解析器中不需要grad，反之亦然，某些参数是编码器参数而在其它参数解析器中不是编码器。

    .. py:method:: values()

        生成所有参数值的迭代器。
