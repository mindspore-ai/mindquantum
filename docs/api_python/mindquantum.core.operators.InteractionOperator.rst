mindquantum.core.operators.InteractionOperator
===============================================

.. py:class:: mindquantum.core.operators.InteractionOperator(constant, one_body_tensor, two_body_tensor)

    存储用于配置费米子分子哈密顿量的“交互算子”的类。哈密顿量包括存储了自旋和奇偶性的单体项和双体项。在这个模块中，存储的系数可以通过费米子算子表示为分子的哈密顿量。

    .. note::
        此类中存储的运算符具有以下形式：

        .. math::

            C + \sum_{p, q} h_{[p, q]} a^\dagger_p a_q +
            \sum_{p, q, r, s} h_{[p, q, r, s]} a^\dagger_p a^\dagger_q a_r a_s.

        其中 :math:`C` 是一个常数。

    参数：
        - **constant** (numbers.Number) - 算子中的常量项，以浮点数形式给出。例如，核排斥能量。
        - **one_body_tensor** (numpy.ndarray) - 单体项的系数（h[p,q]）。这是一个 :math:`n_\text{qubits}\times n_\text{qubits}` 的NumPy浮点数组。默认情况下，存储带有键值的NumPy数组 :math:`a^\dagger_p a_q` (1,0)。
        - **two_body_tensor** (numpy.ndarray) - 双体项的系数 (h[p, q, r, s]) 。这是一个 :math:`n_\text{qubits}\times n_\text{qubits}\times n_\text{qubits}\times n_\text{qubits}` 的NumPy浮点数组。默认情况下，存储带有键值的NumPy数组 :math:`a^\dagger_p a^\dagger_q a_r a_s` (1, 1, 0, 0)。

    .. py:method:: unique_iter(complex_valued=False)

        迭代不在同一对称组中的所有项。

        **四点对称：**

        1. pq = qp。
        2. pqrs = srqp = qpsr = rspq。

        **八点对称（当complex_valued为False时）：**

        1. pq = qp。
        2. pqrs = rqps = psrq = srqp = qpsr = rspq = spqr = qrsp。

        参数：
            - **complex_valued** (bool) - 算子是否有复数系数。默认值： ``False``。
