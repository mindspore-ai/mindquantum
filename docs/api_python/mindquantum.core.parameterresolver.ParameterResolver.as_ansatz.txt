mindquantum.core.parameterresolver.ParameterResolver.as_ansatz()

        将所有参数设置为ansatz。

        返回：
            ParameterResolver，参数解析器本身。

        样例：
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> pr.as_encoder()
            >>> pr.as_ansatz()
            >>> pr.ansatz_parameters
            {'a', 'b'}
'}
