Class mindquantum.core.operators.InteractionOperator(constant, one_body_tensor, two_body_tensor)

    类用于存储用于配置费力子分子哈密顿量的“交互操作符”。

    哈密顿量包括单体和双体项，它们保存了自旋和奇偶性。 
    在这个模块中，存储的系数可以通过费米子算子类表示分子哈密顿量。

    注:
        此类中存储的运算符具有以下形式：

        .. math::

            C + \sum_{p, q} h_{[p, q]} a^\dagger_p a_q +
            \sum_{p, q, r, s} h_{[p, q, r, s]} a^\dagger_p a^\dagger_q a_r a_s.

        其中 :math:`C` 是一个常数。

    参数:
        constant (numbers.Number): 运算符中的常量项，以浮点数形式给出。例如，核排斥能量。
        one_body_tensor (numpy.ndarray): 单体项的系数（h[p,q]）。
            这是一个 :math:`n_\text{qubits}\times n_\text{qubits}` numpy浮点数组。
            默认情况下，我们存储带有键的numpy数组 :math:`a^\dagger_p a_q` (1,0).
        two_body_tensor (numpy.ndarray): 双体项的系数 (h[p, q, r, s]). 
        这是一个 :math:`n_\text{qubits}\times n_\text{qubits}\times n_\text{qubits}\times n_\text{qubits}` nump浮点数组。默认情况下，我们存储带有键的numpy数组 :math:`a^\dagger_p a^\dagger_q a_r a_s` (1, 1, 0, 0).
    