mindquantum.core.gates.gene_univ_parameterized_gate(name, matrix_generator, diff_matrix_generator)

    基于单参数幺正矩阵生成自定义参数化门。

    **参数：**
    - **name** (str) - 此门的名称。
    - **matrix_generator** (Union[FunctionType, MethodType]) - 只采用一个参数生成幺正矩阵的函数或方法。
    - **diff_matrix_generator** (Union[FunctionType, MethodType]) - 只采用一个参数生成幺正矩阵导数的函数或方法。

    **返回：**
        _ParamNonHerm，自定义参数化门。

    **样例：**
        >>> import numpy as np
        >>> from mindquantum import gene_univ_parameterized_gate
        >>> from mindquantum import Simulator, Circuit
        >>> def matrix(theta):
        ...     return np.array([[np.exp(1j * theta), 0],
        ...                      [0, np.exp(-1j * theta)]])
        >>> def diff_matrix(theta):
        ...     return 1j*np.array([[np.exp(1j * theta), 0],
        ...                         [0, -np.exp(-1j * theta)]])
        >>> TestGate = gene_univ_parameterized_gate('Test', matrix, diff_matrix)
        >>> circ = Circuit().h(0)
        >>> circ += TestGate('a').on(0)
        >>> circ
        q0: ──H────Test(a)──
        >>> circ.get_qs(pr={'a': 1.2})
        array([0.25622563+0.65905116j, 0.25622563-0.65905116j])
       