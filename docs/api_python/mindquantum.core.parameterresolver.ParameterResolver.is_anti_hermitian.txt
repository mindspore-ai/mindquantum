mindquantum.core.parameterresolver.ParameterResolver.is_anti_hermitian()

        检查该参数解析器的参数值是否为anti hermitian。

        返回：
            bool，参数解析器是否为anti hermitian。

        样例：
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1})
            >>> pr.is_anti_hermitian()
            False
            >>> (pr + 3).is_anti_hermitian()
            False
            >>> (pr*1j).is_anti_hermitian()
            True
ue
