mindquantum.core.parameterresolver.ParameterResolver.is_hermitian()

        检查该参数解析器的参数值是否为hermitian。

        返回：
            bool，参数解析器是否为hermitian。

        样例：
            >>> from mindquantum.core import ParameterResolver as PR
            >>> pr = PR({'a': 1})
            >>> pr.is_hermitian()
            True
            >>> (pr + 3).is_hermitian()
            True
            >>> (pr * 1j).is_hermitian()
            False
        
  
