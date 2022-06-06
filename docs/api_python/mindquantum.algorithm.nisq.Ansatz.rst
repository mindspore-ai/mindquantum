.. py:class:: mindquantum.algorithm.nisq.Ansatz(name, n_qubits, *args, **kwargs)

    Ansatz的基类。

    **参数：**

    - **name** (str) - ansatz的名字。
    - **n_qubits** (int) - asatz作用于多少个量子位。

    .. py:method:: circuit
        :property:

        获取ansatz的量子电路。
