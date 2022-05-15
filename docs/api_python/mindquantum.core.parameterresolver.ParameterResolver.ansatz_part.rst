mindquantum.core.parameterresolver.ParameterResolver.ansatz_part(*names)

        设置哪个部分是ansatz参数。

        参数：
            names (tuple[str])：将作为一个ansatz.参数。

        返回：
            参数解析器，参数解析器本身。

        样例：
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.as_encoder()
            >>> pr.ansatz_part('a')
            >>> pr.ansatz_parameters
            {'a'}
'}
