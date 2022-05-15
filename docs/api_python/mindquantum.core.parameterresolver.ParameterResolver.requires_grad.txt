mindquantum.core.parameterresolver.ParameterResolver.requires_grad()

        将此参数解析器的所有参数设置为需要梯度计算。直接操作。

        返回：
            ParameterResolver，参数解析器本身。

        样例：
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad_part('a')
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.requires_grad()
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.requires_grad_parameters
            {'a', 'b'}
'}
