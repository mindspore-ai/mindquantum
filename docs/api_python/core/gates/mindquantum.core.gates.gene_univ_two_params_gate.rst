mindquantum.core.gates.gene_univ_two_params_gate
====================================================

.. py:function:: mindquantum.core.gates.gene_univ_two_params_gate(name, matrix_generator, diff_matrix_generator_1, diff_matrix_generator_2)

    基于双参数幺正矩阵生成自定义参数化门。

    .. note::
        矩阵中的元素需要显示的定义为复数，特别是对于多比特门。

    参数：
        - **name** (str) - 此门的名称。
        - **matrix_generator** (Union[FunctionType, MethodType]) - 采用两个参数生成幺正矩阵的函数或方法。
        - **diff_matrix_generator_1** (Union[FunctionType, MethodType]) - 采用两个参数生成幺正矩阵对第一个参数的导数的函数或方法。
        - **diff_matrix_generator_2** (Union[FunctionType, MethodType]) - 采用两个参数生成幺正矩阵对第二个参数的导数的函数或方法。

    返回：
        _TwoParamNonHerm，自定义的双参数化门。