mindquantum.core.parameterresolver.ParameterResolver.conjugate()

        获取参数解析器的共轭。

        返回：
            ParameterResolver，此参数解析器的共轭版本。

        样例：
            >>> from mindquantum.core.parameterresolver import ParameterResolver as PR
            >>> import numpy as np
            >>> pr = PR({'a' : 1, 'b': 1j}, dtype=np.complex128)
            >>> pr.conjugate().expression()
            'a + (-1j)*b'
b'
