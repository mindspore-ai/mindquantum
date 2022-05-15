mindquantum.algorithm.nisq.Transform.parity()

        应用奇偶校验转换。奇偶校验转换非本地存储初始占用数。公式为：

        .. math::

            \left|f_{M−1}, f_{M−2},\cdots, f_0\right> → \left|q_{M−1}, q_{M−2},\cdots, q_0\right>,

        其中

        .. math::

            q_{m} = \left|\left(\sum_{i=0}^{m-1}f_{i}\right) mod\ 2 \right>

        基本上，这个公式可以写成这样，

        .. math::

            p_{i} = \sum{[\pi_{n}]_{i,j}} f_{j},

        其中 :math:'\pi_{n}'是 :math:`N\乘以N`平方矩阵， :math:`N`是总量子位数。运算符的更改遵循以下等式：

        .. math::

            a^\dagger_{j}\rightarrow\frac{1}{2}\left(\prod_{i=j+1}^N
            \left(\sigma_i^X X\right)\right)\left( \sigma^{X}_{j}-i\sigma_j^Y\right) X \sigma^{Z}_{j-1}

            a_{j}\rightarrow\frac{1}{2}\left(\prod_{i=j+1}^N
            \left(\sigma_i^X X\right)\right)\left( \sigma^{X}_{j}+i\sigma_j^Y\right) X \sigma^{Z}_{j-1}

        返回:
            QubitOperator，奇偶校验转换后的量子位运算符。
           