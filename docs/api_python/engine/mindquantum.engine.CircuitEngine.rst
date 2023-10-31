.. py:class:: mindquantum.engine.CircuitEngine

    一个简单的线路引擎，生成projectq格式的量子线路。

    .. py:method:: allocate_qubit()

        分配一个量子比特。

    .. py:method:: allocate_qureg(n)

        分配量子寄存器。

        参数：
            - **n** (int) - 量子比特的数目。

    .. py:method:: circuit
        :property:

        获取这个引擎构造的量子线路。

    .. py:method:: generator(n_qubits, *args, **kwds)
        :staticmethod:

        量子线路寄存器。

        参数：
            - **n_qubits** (int) - 量子线路的量子比特数。
