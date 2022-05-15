mindquantum.core.parameterresolver.ParameterResolver.update(other)

        使用其他参数解析器更新此参数解析器。

        参数：
            others (ParameterResolver)：其他参数解析器。

        异常：
            ValueError：如果某些参数在其他参数解析器中需要grad而不需要grad，反之亦然，某些参数是编码器参数，而不是其他参数解析器中的编码器，反之亦然。

        样例：
            >>> from mindquantum import ParameterResolver
            >>> pr1 = ParameterResolver({'a': 1})
            >>> pr2 = ParameterResolver({'b': 2})
            >>> pr2.no_grad()
            {'b': 2.0}, const: 0.0
            >>> pr1.update(pr2)
            >>> pr1
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr1.no_grad_parameters
            {'b'}
'}
