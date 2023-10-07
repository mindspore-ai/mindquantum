mindquantum.core.operators.PolynomialTensor
============================================

.. py:class:: mindquantum.core.operators.PolynomialTensor(n_body_tensors=None)

    以张量形式存储费米梯算子系数的类。
    例如，在粒子数守恒的分子哈密顿量（4级多项式）中，只有三种项，即常数项、
    单激励 :math:`a^\dagger_p a_q` 和双激励项 :math:`a^\dagger_p a^\dagger_q a_r a_s`，它们对应的系数可以存储在标量、
    :math:`n_\text{qubits}\times n_\text{qubits}` 矩阵和 :math:`n_\text{qubits}\times n_\text{qubits}\times n_\text{qubits}\times n_\text{qubits}` 矩阵中。
    请注意，由于奇偶性守恒，每个张量必须具有偶数维数。
    这个类的大部分功能与FermionOperator的功能相似。

    参数：
        - **n_body_tensors** (dict) - 存储描述n-body作用的张量的字典。
          键是指示张量类型的元组。
          例如， ``n_body_tensors[()]`` 将返回一个常量，
          而 ``n_body_tensors[(1, 0)]`` 将是一个 :math:`n_\text{qubits}\times n_\text{qubits}` 的numpy数组，
          和 ``n_body_tensors[(1,1,0,0)]`` 将返回一个 :math:`n_\text{qubits}\times n_\text{qubits}\times n_\text{qubits}\times n_\text{qubits}` 的numpy数组。
          这些常数和数组分别表示identity、 :math:`a^\dagger_p a_q` 和 :math:`a^\dagger_p a^\dagger_q a_r a_s` 的系数。默认值： ``None``。

    .. note::
        这里的'1'代表 :math:`a^\dagger`，而'0'代表 :math:`a`。

    .. py:method:: constant
      :property:

      获取常数项的值。

    .. py:method:: one_body_tensor
      :property:

      获得单体项。

    .. py:method:: two_body_tensor
      :property:

      获得双体项。
