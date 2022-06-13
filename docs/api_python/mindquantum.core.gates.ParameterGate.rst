.. py:class:: mindquantum.core.gates.ParameterGate(pr: PR, name, n_qubits, *args, obj_qubits=None, ctrl_qubits=None, **kwargs)

    参数化的门。

    .. py:method:: no_grad_part(names)

        设置某些不需要求梯度的参数。此操作将会原位改变线路参数梯度属性。

        **参数：**

        - **names** (tuple[str]) - 不需要求梯度的参数。

        **返回：**

        BasicGate，以及那些不需要更新梯度的参数。

    .. py:method:: requires_grad_part(names)

        设置部分参数需要求导参数。原位操作。

        **参数：**

        - **names** (tuple[str]) - 需要求梯度的参数。

        **返回：**

        BasicGate，返回门本身。
