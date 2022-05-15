mindquantum.core.parameterresolver.ParameterResolver.real

        获取每个参数值的实部。

        返回：
            ParameterResolver，实部件参数值。

        样例：
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR('a', 3) + 1j * PR('a', 4)
            >>> pr
            {'a': (1+1j)}, const: (3+4j)
            >>> pr.real
            {'a': 1.0}, const: 3.0
.0
