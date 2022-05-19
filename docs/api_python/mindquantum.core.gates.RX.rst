.. py:class:: mindquantum.core.gates.RX(coeff=None)

    围绕x轴的旋转门。

    .. math::

        {\rm RX}=\begin{pmatrix}\cos(\theta/2)&-i\sin(\theta/2)\\
                       -i\sin(\theta/2)&\cos(\theta/2)\end{pmatrix}

    该旋转门可以用三种不同的方式初始化。

    1. 如果您用单个数字初始化它，那么它将是一个具有一定旋转角度的非参数化门。

    2. 如果您使用单个str初始化它，那么它将是只有一个参数的参数化门，默认系数为1。

    3. 如果您使用dict初始化它，例如`{'a'：1，'b'：2}'，则此门可以包含多个具有特定系数的参数。在这种情况下，它可以表示为：


    .. math::

        RX(a+2b)

    **参数：**
    - **coeff** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。默认值：None。

    **样例：**
        >>> from mindquantum.core.gates import RX
        >>> import numpy as np
        >>> rx1 = RX(0.5)
        >>> np.round(rx1.matrix(), 2)
        array([[0.97+0.j  , 0.  -0.25j],
               [0.  -0.25j, 0.97+0.j  ]])
        >>> rx2 = RX('a')
        >>> np.round(rx2.matrix({'a':0.1}), 3)
        array([[0.999+0.j  , 0.   -0.05j],
               [0.   -0.05j, 0.999+0.j  ]])
        >>> rx3 = RX({'a' : 0.2, 'b': 0.5}).on(0, 2)
        >>> print(rx3)
        RX(0.2*a + 0.5*b|0 <-: 2)
        >>> np.round(rx3.matrix({'a' : 1, 'b' : 2}), 2)
        array([[0.83+0.j  , 0.  -0.56j],
               [0.  -0.56j, 0.83+0.j  ]])
        >>> np.round(rx3.diff_matrix({'a' : 1, 'b' : 2}, about_what = 'a'), 2)
        array([[-0.06+0.j  ,  0.  -0.08j],
               [ 0.  -0.08j, -0.06+0.j  ]])
        >>> rx3.coeff
        {'a': 0.2, 'b': 0.5}
    