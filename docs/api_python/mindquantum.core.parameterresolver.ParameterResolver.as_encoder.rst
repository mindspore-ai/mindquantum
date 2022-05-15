mindquantum.core.parameterresolver.ParameterResolver.as_encoder()

        将所有参数设置为编码器。

        返回：
            ParameterResolver，参数解析器本身。

        样例：
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1, 'b': 2})
            >>> pr.as_encoder()
            >>> pr.encoder_parameters
            {'a', 'b'}
'}
