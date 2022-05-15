mindquantum.core.parameterresolver.ParameterResolver.requires_grad_parameters

        获取需要梯度的参数。

        返回：
            set，需要梯度的参数集。

        >>> from mindquantum.core import ParameterResolver as PR
        >>> a = PR({'a': 1, 'b': 2})
        >>> a.requires_grad_parameters
        {'a', 'b'}
'}
