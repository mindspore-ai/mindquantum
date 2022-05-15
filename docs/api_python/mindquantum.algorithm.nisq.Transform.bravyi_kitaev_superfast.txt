mindquantum.algorithm.nisq.Transform.bravyi_kitaev_superfast()

        应用Bravyi-Kitaev超快变换。
        从https://arxiv.org/pdf/1712.00446.pdf实施

        请注意，只有形式的隐数运算符

        .. math::

            C + \sum_{p, q} h_{p, q} a^\dagger_p a_q +
                \sum_{p, q, r, s} h_{p, q, r, s} a^\dagger_p a^\dagger_q a_r a_s

        其中 :math:`C`是一个常数，进行转换。

        返回:
            量子位运算符，在bravyi_itaev_超快之后的量子位运算符。
        