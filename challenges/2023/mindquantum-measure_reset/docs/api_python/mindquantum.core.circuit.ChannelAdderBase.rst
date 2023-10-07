mindquantum.core.circuit.ChannelAdderBase
=========================================

.. py:class:: mindquantum.core.circuit.ChannelAdderBase(add_after=True)

    在量子门前面或者后面添加信道。

    本方法为所有信道添加器的基类。在所有的派生类中，你需要定义 `_accepter`、 `_excluder` 和  `_handler` 方法。 `_accepter` 方法是一些接受规则的集合，每一个想要添加噪声的量子门都需要满足这些接受规则。 `_excluder` 方法是一些拒绝规则的集合，每一个想要添加噪声的量子信道都需要不被这些规则接受。 `_handler` 是在满足接受规则和拒绝拒绝规则时对量子门的具体操作。

    参数：
        - **add_after** (bool) - 在量子门前面或者后面添加量子信道。默认值： ``True``。
