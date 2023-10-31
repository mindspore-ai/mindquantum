mindquantum.algorithm.nisq.Ansatz
==================================

.. py:class:: mindquantum.algorithm.nisq.Ansatz(name, n_qubits, *args, **kwargs)

    Ansatz的基类。

    参数：
        - **name** (str) - ansatz的名字。
        - **n_qubits** (int) - ansatz作用于多少个量子比特。

    .. py:method:: circuit
        :property:

        获取ansatz的量子线路。

        返回：
            Circuit，ansatz的量子线路。
