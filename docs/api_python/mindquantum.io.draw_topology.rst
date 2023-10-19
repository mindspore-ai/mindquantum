mindquantum.io.draw_topology
============================

.. py:function:: mindquantum.io.draw_topology(topo: QubitsTopology, circuit: Circuit = None, style: Dict = None, edge_color: Union[str, Dict[Tuple[int, int], str]] = None)

    以svg图的形式打印量子拓扑结构。

    参数：
        - **topo** (:class:`.device.QubitsTopology`) - 量子比特拓扑结构。
        - **circuit** (:class:`~.core.circuit.Circuit`) - 想要在指定拓扑结构上执行的量子线路。默认值： ``None``。
        - **style** (Dict) - 绘制的格式配置字典。默认值： ``None``。
        - **edge_color** (Union[str, Dict[Tuple[int, int], str]]) - 边的颜色。如果为颜色字符串，则图中每条边都以给定颜色着色。也可输入字典指定不同边的着色。其中字典的键为边，值为颜色字符串。
