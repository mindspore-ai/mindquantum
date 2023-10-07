mindquantum.core.gates.ParameterGate
=====================================

.. py:class:: mindquantum.core.gates.ParameterGate(pr: ParameterResolver, name, n_qubits, *args, obj_qubits=None, ctrl_qubits=None, **kwargs)

    参数化的门。

    参数：
        - **pr** (ParameterResolver) - 参数化量子门的参数。
        - **name** (str) - 参数化量子门的名字。
        - **n_qubits** (int) - 参数化量子门的比特数。
        - **args** (list) - 量子门的其他参数。
        - **obj_qubits** (Union[int, List[int]]) - 量子门作用在哪些比特上。默认值： ``None`` 。
        - **ctrl_qubits** (Union[int, List[int]]) - 量子门受哪些量子比特控制。默认值： ``None`` 。
        - **kwargs** (dict) - 量子门的其他参数。

    .. py:method:: get_parameters()

        返回参数化门的参数列表。

    .. py:method:: no_grad()

        设置量子门中的所有参数都不需要求导数。

    .. py:method:: no_grad_part(names)

        设置某些不需要求梯度的参数。此操作将会原位改变线路参数梯度属性。

        参数：
            - **names** (tuple[str]) - 不需要求梯度的参数。

        返回：
            BasicGate，其中有些参数不需要更新梯度。

    .. py:method:: requires_grad()

        设置量子门中的所有参数都需要求导数。

    .. py:method:: requires_grad_part(names)

        设置哪部分参数需要求导。原地操作。

        参数：
            - **names** (tuple[str]) - 需要求梯度的参数。

        返回：
            BasicGate，其中有些参数需要更新梯度。
