mindquantum.core.gates.gene_univ_parameterized_gate
====================================================

.. py:function:: mindquantum.core.gates.gene_univ_parameterized_gate(name, matrix_generator, diff_matrix_generator)

    基于单参数幺正矩阵生成自定义参数化门。

    参数：
        - **name** (str) - 此门的名称。
        - **matrix_generator** (Union[FunctionType, MethodType]) - 只采用一个参数生成幺正矩阵的函数或方法。
        - **diff_matrix_generator** (Union[FunctionType, MethodType]) - 只采用一个参数生成幺正矩阵导数的函数或方法。

    返回：
        _ParamNonHerm，自定义的参数化门。
