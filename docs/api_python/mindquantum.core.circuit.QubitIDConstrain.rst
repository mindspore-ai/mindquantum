mindquantum.core.circuit.QubitIDConstrain
=========================================

.. py:class:: mindquantum.core.circuit.QubitIDConstrain(adder: ChannelAdderBase)

    只将噪声信道作用在给定比特序号的量子门上。

    参数：
        - **qubit_ids** (Union[int, List[int]]) - 想要选择的比特序号的列表。
        - **add_after** (bool) - 在量子门前面或者后面添加量子信道。默认值： ``True``。
