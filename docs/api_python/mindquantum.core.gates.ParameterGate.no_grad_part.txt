mindquantum.core.gates.ParameterGate.no_grad_part(names)

        设置某些不需要渐变的参数。就地操作。

        参数:
            names (tuple[str]): 不需要梯度的参数。

        返回:
            BasicGate，部分参数不需要更新梯度。
        