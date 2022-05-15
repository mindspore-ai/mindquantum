mindquantum.core.parameterresolver.ParameterResolver.is_const()

        检查此参数解析器是否表示常量，这意味着此参数解析器中没有非零参数。

        返回：
            bool，此参数解析器是否表示常量数。

        样例：
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR(1.0)
            >>> pr.is_const()
            Truerue