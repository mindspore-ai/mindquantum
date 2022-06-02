.. py:method:: mindquantum.algorithm.nisq.Transform.bravyi_kitaev_superfast()

    作用快速Bravyi-Kitaev变换。
    基于 https://arxiv.org/pdf/1712.00446.pdf 实现。

    请注意，只有如下的厄密共轭算符才能进行转换。

    .. math::

        C + \sum_{p, q} h_{p, q} a^\dagger_p a_q +
            \sum_{p, q, r, s} h_{p, q, r, s} a^\dagger_p a^\dagger_q a_r a_s

    其中 :math:`C` 是一个常数。

    **返回：**

    QubitOperator，经过快速bravyi_kitaev变换之后的玻色子算符。
