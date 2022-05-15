mindquantum.core.gates.ParameterGate.requires_grad_part(names)

        设置某些需要渐变的参数。就地操作。

        参数:
            names (tuple[str]): 需要梯度的参数。

        返回:
            BasicGate，部分参数需要更新梯度。
        