mindquantum.core.parameterresolver.ParameterResolver.expression()

        获取此参数解析器的表达式字符串。

        返回：
            str，此参数解析器的字符串表达式。

        样例：
            >>> from mindquantum.core import ParameterResolver as PR
            >>> import numpy as np
            >>> pr = PR({'a': np.pi}, np.sqrt(2))
            >>> pr.expression()
            'π*a + √2'
2'
