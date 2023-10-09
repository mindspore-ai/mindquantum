mindquantum.device.QubitsTopology
=================================

.. py:class:: mindquantum.device.QubitsTopology(qubits: typing.List[QubitNode])

    量子比特在硬件设备上的拓扑结构图。

    拓扑结构是由不同的 :class:`~.device.QubitNode` 构成，并且你可以直接设置每一个量子比特的信息。

    参数：
        - **qubits** (List[:class:`~.device.QubitNode`]) - 拓扑结构中的所有量子比特。

    .. py:method:: add_qubit_node(qubit: QubitNode)

        在拓扑结构中添加一个量子比特。

        参数：
            - **qubit** (:class:`~.device.QubitNode`) - 想要添加到拓扑结构中的量子比特。

    .. py:method:: all_qubit_id()

        获取所有比特的 id 信息。

        返回：
            Set[int]，所有比特的 id。

    .. py:method:: choose(ids: typing.List[int]) -> typing.List[QubitNode]:

        根据给定的 id 选择量子比特。

        参数：
            - **ids** (List[int]) - 一个量子比特 id 的列表。

        返回：
            List[:class:`~.device.QubitNode`]，根据给定 id 选择出的量子比特列表。

    .. py:method:: edges_with_id() -> typing.Set[typing.Tuple[int, int]]

        返回用 id 表示的图中的边。

        返回：
            Set[Tuple[int, int]]，量子拓扑结构中相连接的量子比特所在的边。

    .. py:method:: edges_with_poi() -> typing.Set[typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]

        返回用坐标表示的图中的边。

        返回：
            Set[Tuple[Tuple[float, float], Tuple[float, float]]]，量子拓扑结构中相连接的量子比特所在的边，用坐标表示。

    .. py:method:: has_qubit_node(qubit_id: int) -> bool:

        检查某个量子比特是否在该拓扑结构中。

        参数：
            - **qubit_id** (int) - 想要检查的量子比特的 id。

        返回：
            bool，当前拓扑结构是否拥有给定 id 的比特。

    .. py:method:: is_coupled_with(id1: int, id2: int) -> bool:

        检查两个比特是否联通，也即是否有耦合。

        参数：
            - **id1** (int) - 第一个比特的 id。
            - **id2** (int) - 另外一个比特的 id。

        返回：
            bool，给定的两个比特是否联通。

    .. py:method:: isolate_with_near(qubit_id: int) -> None:

        将给定比特与相连接的比特解耦。

        参数：
            - **qubit_id** (int) - 需要解耦的比特的 id。

    .. py:method:: n_edges() -> int:

        获取所有有耦合的边的个数。

        返回：
            int，拓扑结构中有耦合的边的个数。

    .. py:method:: remove_isolate_node() -> None:

        移除那些不与其他比特有耦合的比特。

    .. py:method:: remove_qubit_node(qubit_id: int) -> None:

        移除一个给定的比特。

        参数：
            - **qubit_id** (int) - 想要移除的那个比特。

    .. py:method:: select(ids: typing.List[int]) -> "QubitsTopology":

        选择一些比特节点并生成新的拓扑图。

        参数：
            - **ids** (List[int]) - 比特节点id的列表。

        返回：
            :class:`~.device.QubitsTopology`，保持连接信息的新的拓扑图。

    .. py:method:: set_color(qubit_id: int, color: str) -> None:

        设置给定比特的颜色。

        参数：
            - **qubit_id** (int) - 想要改变颜色的量子别的 id。
            - **color** (str) - RGB颜色。

    .. py:method:: set_position(qubit_id: int, poi_x: float, poi_y: float) -> None:

        设置给定比特的位置。

        参数：
            - **qubit_id** (int) - 想要改变位置的量子比特的 id。
            - **poi_x** (float) - 新的 x 轴坐标。
            - **poi_y** (float) - 新的 y 轴坐标。

    .. py:method:: size() -> int:

        获得总比特数。

        返回：
            int，总的比特数。
