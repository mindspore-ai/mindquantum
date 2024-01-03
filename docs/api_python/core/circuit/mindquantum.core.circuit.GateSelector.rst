mindquantum.core.circuit.GateSelector
=====================================

.. py:class:: mindquantum.core.circuit.GateSelector(gate: str)

    挑选量子门来添加噪声信道。

    参数：
        - **gate** (str) - 想要添加信道的量子门。当前可以是 'H'，'X'，'Y', 'Z'，'RX'，'RY'，'RZ'，'CX'，'CZ'，'SWAP'。

    .. py:method:: supported_gate()
        :property:

        获取门选择器支持的门。
