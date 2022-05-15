mindquantum.core.parameterresolver.ParameterResolver.imag

        获取每个参数值的图像部分。

        返回：
            ParameterResolver，图像部分参数值。

        样例：
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR('a', 3) + 1j * PR('a', 4)
            >>> pr
            {'a': (1+1j)}, const: (3+4j)
            >>> pr.imag
            {'a': 1.0}, const: 4.0
.0
