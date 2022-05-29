mindquantum.core.gates.ParameterGate.requires_grad_part(names)

        设置某些需要求梯度的参数。Inplace operation.

        **参数：**
        - **names** (tuple[str]) - 需要求梯度的参数。

        **返回：**
            BasicGate，以及那些需要更新梯度的参数。
        