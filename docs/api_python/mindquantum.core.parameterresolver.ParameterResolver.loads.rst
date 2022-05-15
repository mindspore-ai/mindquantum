mindquantum.core.parameterresolver.ParameterResolver.loads(strs)

        将JSON（JavaScript对象表示法）加载到FermionOperator中

        参数：
            strs (str)：转储参数解析器字符串。

        返回：
            FermionOperator，从字符串加载的FermionOperator

        样例：
            >>> from mindquantum.core.parameterresolver import ParameterResolver
            >>> strings = """
            ...     {
            ...         "a": 1,
            ...         "b": 2,
            ...         "c": 3,
            ...         "d": 4,
            ...         "__class__": "ParameterResolver",
            ...         "__module__": "parameterresolver",
            ...         "no_grad_parameters": [
            ...             "a",
            ...             "b"
            ...         ]
            ...     }
            ...     """
            >>> obj = ParameterResolver.loads(string)
            >>> print(obj)
            {'a': 1, 'b': 2, 'c': 3, 'd': 4}
            >>> print('requires_grad_parameters is:', obj.requires_grad_parameters)
            requires_grad_parameters is: {'c', 'd'}
            >>> print('no_grad_parameters is :', obj.no_grad_parameters)
            no_grad_parameters is : {'b', 'a'}a'}