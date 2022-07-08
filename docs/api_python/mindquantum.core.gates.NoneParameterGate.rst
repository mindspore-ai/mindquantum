.. py:class:: mindquantum.core.gates.NoneParameterGate(name, n_qubits, obj_qubits=None, ctrl_qubits=None)

    非参数化的门。

    **参数：**

    - **name** (str) - 参数化量子门的名字。
    - **n_qubits** (int) - 参数化量子门的比特数。
    - **obj_qubits** (Union[int, List[int]]) - 量子门作用在哪些比特上。默认值： `None` 。
    - **ctrl_qubits** (Union[int, List[int]]) - 量子门受哪些量子比特控制。默认值： `None` 。
