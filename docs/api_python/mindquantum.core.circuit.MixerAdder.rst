mindquantum.core.circuit.MixerAdder
===================================

.. py:class:: mindquantum.core.circuit.MixerAdder(adders: typing.List[ChannelAdderBase], add_after=True)

    在子添加器的接受集被满足、拒绝集被拒绝时依次执行所有的添加器。

    参数：
        - **adders** (List[:class:`~.core.gates.BitFlipChannel`]) - 想要混合的噪声添加器。
        - **add_after** (bool) - 是否在量子门后面添加信道。如果为 ``False``，信道将会加在量子门前面。默认值： ``True``。
