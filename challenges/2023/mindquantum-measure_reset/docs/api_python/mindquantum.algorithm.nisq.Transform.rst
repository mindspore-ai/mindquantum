mindquantum.algorithm.nisq.Transform
=====================================

.. py:class:: mindquantum.algorithm.nisq.Transform(operator, n_qubits=None)

    将费米子或者玻色子进行转化的模块。
    `jordan_wigner` , `parity` , `bravyi_kitaev` , `bravyi_kitaev_tree` , `bravyi_kitaev_superfast` 将会把 `FermionOperator` 转换为 `QubitOperator`。 `reversed_jordan_wigner` 将会把 `QubitOperator` 转换为 `FermionOperator` 。

    参数：
        - **operator** (Union[FermionOperator, QubitOperator]) - 需要进行转换的 `FermionOperator` 或 `QubitOperator` 。
        - **n_qubits** (int) - 输入算符的比特数。如果为 ``None`` ， 系统将会自动数出比特数。默认值： ``None``。

    .. py:method:: bravyi_kitaev()

        进行Bravyi-Kitaev变换。

        Bravyi-Kitaev是介于Jordan-Wigner变换和parity变换之间的变换。也就是说，它平衡了占据的局部性和宇称信息，以提高模拟效率。在此方案中，量子比特存储一组 :math:`2^x` 轨道的宇称，其中 :math:`x \ge 0` 。索引 :math:`j` 的量子比特总是存储轨道 :math:`j` 。对于偶数的 :math:`j` ，这是它存储的唯一轨道。但对于奇数的 :math:`j` ，它还存储索引小于 :math:`j` 的一组相邻轨道。
        对于占据态变换，我们遵循公式：

        .. math::

            b_{i} = \sum{[\beta_{n}]_{i,j}} f_{j},

        其中 :math:`\beta_{n}` 是 :math:`N\times N` 维平方矩阵， :math:`N` 是总量子数。量子比特的索引分为三个集合，宇称集、更新集和翻转集。这组量子比特的宇称与索引小于 :math:`j` 的轨道集具有相同的宇称，因此我们将称这组量子比特索引为“宇称集” :math:`j` ，或 :math:`P(j)` 。

        索引为 :math:`j` 的更新集，或 :math:`U(j)` 包含除序号为 :math:`j` 的量子比特会被更新，当轨道 :math:`j` 被占据时。
        索引为 :math:`j` 的翻转集，或 :math:`F(j)` ，包含所有的BravyiKitaev量子比特，这些比特将决定量子比特 :math:`j` 相对于轨道 :math:`j` 来说是否有相同或者相反的宇称。

        请参见论文中的一些详细解释 `The Bravyi-Kitaev transformation for quantum computation of electronic structure <https://doi.org/10.1063/1.4768229>`_。

        本方法基于 `Fermionic quantum computation <https://arxiv.org/abs/quant-ph/0003137>`_ 和 `A New Data Structure for Cumulative Frequency Tables <https://doi.org/10.1002/spe.4380240306>`_ 实现。

        返回：
            QubitOperator，经过 `bravyi_kitaev` 变换的玻色子算符。

    .. py:method:: bravyi_kitaev_superfast()

        作用快速Bravyi-Kitaev变换。
        基于 `Bravyi-Kitaev Superfast simulation of fermions on a quantum computer <https://arxiv.org/abs/1712.00446>`_ 实现。

        请注意，只有如下的厄米共轭算符才能进行转换。

        .. math::

            C + \sum_{p, q} h_{p, q} a^\dagger_p a_q +
                \sum_{p, q, r, s} h_{p, q, r, s} a^\dagger_p a^\dagger_q a_r a_s

        其中 :math:`C` 是一个常数。

        返回：
            QubitOperator，经过快速bravyi_kitaev变换之后的玻色子算符。

    .. py:method:: jordan_wigner()

        应用Jordan-Wigner变换。Jordan-Wigner变换能够保留初始占据数的局域性，并按照如下的形式将费米子转化为玻色子。

        .. math::

            a^\dagger_{j}\rightarrow \sigma^{-}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i}

            a_{j}\rightarrow \sigma^{+}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i},

        其中 :math:`\sigma_{+} = \sigma^{X} + i \sigma^{Y}` 和 :math:`\sigma_{-} = \sigma^{X} - i\sigma^{Y}` 分别是自旋升算符和降算符。

        返回：
            QubitOperator，Jordan-Wigner变换后的量子比特算符。

    .. py:method:: parity()

        应用宇称变换。宇称变换保存初始占据数的非局域性。公式为：

        .. math::

            \left|f_{M-1}, f_{M-2},\cdots, f_0\right> \rightarrow \left|q_{M-1}, q_{M-2},\cdots, q_0\right>,

        其中

        .. math::

            q_{m} = \left|\left(\sum_{i=0}^{m-1}f_{i}\right) mod\ 2 \right>

        该公式可以写成这样，

        .. math::

            p_{i} = \sum{[\pi_{n}]_{i,j}} f_{j},

        其中 :math:`\pi_{n}` 是 :math:`N\times N` 维的方矩阵， :math:`N` 是总量子比特数。运算按照如下等式进行：

        .. math::

            a^\dagger_{j}\rightarrow\frac{1}{2}\left(\prod_{i=j+1}^N
            \left(\sigma_i^X X\right)\right)\left( \sigma^{X}_{j}-i\sigma_j^Y\right) X \sigma^{Z}_{j-1}

            a_{j}\rightarrow\frac{1}{2}\left(\prod_{i=j+1}^N
            \left(\sigma_i^X X\right)\right)\left( \sigma^{X}_{j}+i\sigma_j^Y\right) X \sigma^{Z}_{j-1}

        返回：
            QubitOperator，经过宇称变换后的玻色子算符。

    .. py:method:: reversed_jordan_wigner()

        应用Jordan-Wigner逆变换。

        返回：
            FermionOperator，Jordan-Wigner逆变换后的费米子算符。

    .. py:method:: ternary_tree()

        作用Ternary tree变换。
        基于 `Optimal fermion-to-qubit mapping via ternary trees with applications to reduced quantum states learning <https://arxiv.org/abs/1910.10746>`_ 实现。

        返回：
            QubitOperator，Ternary tree变换后的玻色子算符。
