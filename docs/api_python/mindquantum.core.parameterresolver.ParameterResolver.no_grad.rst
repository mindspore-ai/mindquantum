mindquantum.core.parameterresolver.ParameterResolver.no_grad()

        将所有参数设置为不需要计算梯度。直接操作。

        返回：
            ParameterResolver，参数解析器本身。

        样例：
            >>> from mindquantum import ParameterResolver
            >>> pr = ParameterResolver({'a': 1, 'b': 2})
            >>> pr.no_grad()
            {'a': 1.0, 'b': 2.0}, const: 0.0
            >>> pr.requires_grad_parameters
            set()t()