.. py:method:: mindquantum.core.gates.ParameterGate.no_grad_part(names)

    设置某些不需要求梯度的参数。此操作将会原位改变线路参数梯度属性。

    **参数：**

    - **names** (tuple[str]) - 不需要求梯度的参数。

    **返回：**

    BasicGate，以及那些不需要更新梯度的参数。
        