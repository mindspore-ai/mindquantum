Class mindquantum.core.operators.PolynomialTensor(n_body_tensors=None)

    类以张量形式存储费米梯算子的系数。
    例如，在守恒粒子数的分子哈密顿量（4级多项式）中，只有三种项，即常数项，
    单激励 :math:`a^\dagger_p a_q` 和双激励项 :math:`a^\dagger_p a^\dagger_q a_r a_s`, 它们对应的系数可以存储在标量中，
    :math:`n_\text{qubits}\times n_\text{qubits}` 矩阵和 :math:`n_\text{qubits}\times n_\text{qubits} n_\text{qubits}\times n_\text{qubits}` 矩阵。
    请注意，由于奇偶校验守恒，每个张量必须具有偶数维数。
    这个类的大部分功能与FermionOperator的功能相似。

    参数:
        n_body_tensors(dict): 存储描述n-body交互的张量的字典。
            键是指示张量类型的元组。
            例如，n_body_tensors[()]将返回一个常量，
            而n_body_tensors[(1, 0)]将是一个 :math:`n_\text{qubits}\times n_\text{qubits}` numpy数组，
            和n_body_tensors[(1,1,0,0)]将返回一个 :math:`n_\text{qubits}\times n_\text{qubits} n_\text{qubits}\times n_\text{qubits}` numpy数组
            这些常数和数组分别表示形式恒等式的项系数，:math:`a^\dagger_p a_q`, :math:`a^\dagger_p a^\dagger_q a_r a_s`, respectively.。默认值：None。

    注:
        这里的“1”代表 :math:`a^\dagger`, 而“0”代表 :math:`a`。

    样例:
        >>> import numpy as np
        >>> from mindquantum.core.operators import PolynomialTensor
        >>> constant = 1
        >>> one_body_term = np.array([[1,0],[0,1]])
        >>> two_body_term = two_body_term = np.array([[[[1,0],[0,1]],[[1,0],[0,1]]],[[[1,0],[0,1]],[[1,0],[0,1]]]])
        >>> n_body_tensors = {(): 1, (1,0): one_body_term,(1,1,0,0):two_body_term}
        >>> poly_op = PolynomialTensor(n_body_tensors)
        >>> poly_op
        () 1
        ((0, 1), (0, 0)) 1
        ((1, 1), (1, 0)) 1
        ((0, 1), (0, 1), (0, 0), (0, 0)) 1
        ((0, 1), (0, 1), (1, 0), (1, 0)) 1
        ((0, 1), (1, 1), (0, 0), (0, 0)) 1
        ((0, 1), (1, 1), (1, 0), (1, 0)) 1
        ((1, 1), (0, 1), (0, 0), (0, 0)) 1
        ((1, 1), (0, 1), (1, 0), (1, 0)) 1
        ((1, 1), (1, 1), (0, 0), (0, 0)) 1
        ((1, 1), (1, 1), (1, 0), (1, 0)) 1
        >>> # get the constant
        >>> poly_op.constant
        1
        >>> # set the constant
        >>> poly_op.constant = 2
        >>> poly_op.constant
        2
        >>> poly_op.n_qubits
        2
        >>> poly_op.one_body_tensor
        array([[1, 0],
               [0, 1]])
        >>> poly_op.two_body_tensor
        array([[[[1, 0],
                 [0, 1]],
                [[1, 0],
                 [0, 1]]],
               [[[1, 0],
                 [0, 1]],
                 [[1, 0],
                  [0, 1]]]])
    