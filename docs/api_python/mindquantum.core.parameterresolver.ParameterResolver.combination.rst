mindquantum.core.parameterresolver.ParameterResolver.combination(other)

        在此参数解析器与输入pr之间应用线性组合。

        参数：
            pr (Union[dict, ParameterResolver])：参数解析器，想做线性组合。

        返回：
            数字。Number，组合结果。

        样例：
            >>> from mindquantum import ParameterResolver
            >>> pr1 = ParameterResolver({'a': 1, 'b': 2})
            >>> pr2 = ParameterResolver({'a': 2, 'b': 3})
            >>> pr1.combination(pr2)
            {}, const: 8.0
.0
