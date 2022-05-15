mindquantum.core.parameterresolver.ParameterResolver.encoder_part(*names)

        设置哪一部分是编码器参数。

        参数：
            names (tuple[str])：将用作编码器的参数。

        返回：
            ParameterResolver，参数解析器本身。

        样例：
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.encoder_part('a')
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.encoder_parameters
            {'a'}
'}
