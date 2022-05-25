.. py:class:: mindquantum.core.parameterresolver.ParameterResolver(data=None, const=None, dtype=<class 'numpy.float64'>)

    ParameterRsolver可以设置参数化量子门或参数化量子电路的参数。

    根据具体哪一部分参数需要计算梯度，PQC算子只能计算这部分参数的梯度。

    **参数：**

    - **data** (Union[dict, numbers.Number, str, ParameterResolver]) – 初始参数名称及其值。如果数据是dict，则键将是参数名称，值将是参数值。如果数据是数字，则此数字将是此参数解析器的常量值。如果数据是字符串，则此字符串将是系数为1的唯一参数。默认值：无。
    - **const** (number.Number) – 此参数解析器的常量部分。默认值：无。
    - **dtype** (type) – 此参数解析器的值类型。默认值：numpy.float64。
