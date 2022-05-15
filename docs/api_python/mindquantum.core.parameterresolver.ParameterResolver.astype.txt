mindquantum.core.parameterresolver.ParameterResolver.astype(dtype, inplace=False)

        更改此参数解析器的数据类型。

        参数：
            dtype (type)：数据类型。
            inplace (bool)：是否在位更改类型。

        返回：
            ParameterResolver，具有给定数据类型的参数解析器。

        样例：
            >>> from mindquantum.core import ParameterResolver as PR
            >>> import numpy as np
            >>> pr = PR({'a': 1.0}, 2.0)
            >>> pr
            {'a': 1.0}, const: 2.0
            >>> pr.astype(np.complex128, inplace=True)
            >>> pr
            {'a': (1+0j)}, const: (2+0j)
j)
