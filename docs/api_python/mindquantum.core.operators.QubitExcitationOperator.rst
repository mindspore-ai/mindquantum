Class mindquantum.core.operators.QubitExcitationOperator(term=None, coefficient=1.0)

    Qubit激励运算符定义为：
    :math:`Q^{\dagger}_{n} = \frac{1}{2} (X_{n} - iY_{n})` 和 :math:`Q_{n} = \frac{1}{2} (X_{n} + iY_{n})`。
    与费米子激励算子相比，Qubit激励算子是某种“局部化”，即，
    费米子激励运算符 :math:`a^{\dagger}_{7} a_{0}` 涉及JW变换下的量子位范围为0到7，而量子位激励 :math:`Q^{\dagger}_{7} Q_{0}` 将仅影响第0和第7个量子位。
    此外，使用Qubit激励算子描述的双激励比相应的费米子激励算子使用的CNOT要少得多。

    参数:
        terms (str): 量子位激励运算符的输入项。默认值：None。
        coefficient (Union[numbers.Number, str, ParameterResolver]): 对应单运算符的系数，默认值：1.0。

    样例:
        >>> from mindquantum.algorithm.nisq.chem import Transform
        >>> from mindquantum.core.operators import QubitExcitationOperator
        >>> op = QubitExcitationOperator(((4, 1), (1, 0), (0, 0)), 2.5)
        >>> op
        2.5 [Q4^ Q1 Q0]
        >>> op.fermion_operator
        2.5 [4^ 1 0]
        >>> op.to_qubit_operator()
        0.3125 [X0 X1 X4] +
        -0.3125j [X0 X1 Y4] +
        0.3125j [X0 Y1 X4] +
        (0.3125+0j) [X0 Y1 Y4] +
        0.3125j [Y0 X1 X4] +
        (0.3125+0j) [Y0 X1 Y4] +
        (-0.3125+0j) [Y0 Y1 X4] +
        0.3125j [Y0 Y1 Y4]
        >>> Transform(op.fermion_operator).jordan_wigner()
        (0.3125+0j) [X0 X1 Z2 Z3 X4] +
        -0.3125j [X0 X1 Z2 Z3 Y4] +
        0.3125j [X0 Y1 Z2 Z3 X4] +
        (0.3125+0j) [X0 Y1 Z2 Z3 Y4] +
        0.3125j [Y0 X1 Z2 Z3 X4] +
        (0.3125+0j) [Y0 X1 Z2 Z3 Y4] +
        (-0.3125+0j) [Y0 Y1 Z2 Z3 X4] +
        0.3125j [Y0 Y1 Z2 Z3 Y4]
    