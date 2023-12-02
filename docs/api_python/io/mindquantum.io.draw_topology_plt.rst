mindquantum.io.draw_topology_plt
================================

.. py:function:: mindquantum.io.draw_topology_plt(topo: QubitsTopology, circuit: Circuit = None, style: Dict = None)

    用 matplotlib 打印量子拓扑结构。

    参数：
        - **topo** (:class:`.device.QubitsTopology`) - 量子比特拓扑结构。
        - **circuit** (:class:`~.core.circuit.Circuit`) - 想要在指定拓扑结构上执行的量子线路。默认值： ``None``。
        - **style** (Dict) - 绘制的格式配置字典。默认值： ``None``。
