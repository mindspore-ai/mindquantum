mindquantum.core.gates.gene_univ_parameterized_gate
====================================================

.. py:function:: mindquantum.core.gates.gene_univ_parameterized_gate(name, matrix_generator, diff_matrix_generator)

    基于单参数幺正矩阵生成自定义参数化门。

    .. note::
        矩阵中的元素需要显示的定义为复数，特别时对于多比特门。

    参数：
        - **name** (str) - 此门的名称。
        - **matrix_generator** (Union[FunctionType, MethodType]) - 只采用一个参数生成幺正矩阵的函数或方法。
        - **diff_matrix_generator** (Union[FunctionType, MethodType]) - 只采用一个参数生成幺正矩阵导数的函数或方法。

    返回：
        _ParamNonHerm，自定义的参数化门。
