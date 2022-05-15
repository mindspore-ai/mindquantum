mindquantum.core.parameterresolver.ParameterResolver.no_grad_part(*names)

        设置部分不需要梯度的参数。

        参数：
            names (tuple[str])：不需要梯度的参数。

        返回：
            ParameterResolver，参数解析器本身。

        样例：
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad_part('a')
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.requires_grad_parameters
            {'b'}
'}
