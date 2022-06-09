.. py:class:: mindquantum.algorithm.nisq.Ansatz(name, n_qubits, *args, **kwargs)

    Ansatz的基类。

    **参数：**

    - **name** (str) - ansatz的名字。
    - **n_qubits** (int) - ansatz作用于多少个量子位。

    .. py:method:: circuit
        :property:

        获取ansatz量子电路。

        **返回：**
 
        Circuit，ansatz量子电路。
