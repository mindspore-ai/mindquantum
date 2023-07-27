mindquantum.device.QubitNode
============================

.. py:class:: mindquantum.device.QubitNode(qubit_id: int, color: str = '#000000', poi_x: float = 0.0, poi_y: float = 0.0)

    量子比特节点。

    一个量子比特节点拥有一个 id 信息，一个位置信息，和一个颜色信息（如果想绘制量子比特节点）。你可以利用 '>>' 和 '<<' 运算符来连接两个比特，用 '>' 和 '<' 来打断两个比特。

    参数：
        - **qubit_id** (int) - 量子比特节点的 id。
        - **color** (str) - 量子比特的颜色。
        - **poi_x** (float) - 量子比特在绘制平面上的 x 坐标。
        - **poi_y** (float) - 量子比特在绘制平面上的 y 坐标。

    .. py:method:: color
        :property:

        获取比特的颜色信息。

        返回：
            str，量子比特的颜色。

    .. py:method:: poi_x
        :property:

        获取比特的 x 坐标。

        返回：
            float，量子比特的 x 坐标。

    .. py:method:: poi_y
        :property:

        获取比特的 y 坐标。

        返回：
            float，量子比特的 y 坐标。

    .. py:method:: qubit_id
        :property:

        获取比特的 id 信息。

        返回：
            int，量子比特的 id。

    .. py:method:: set_color(color:str)

        设置量子比特的颜色。

        参数：
            - **color** (str) - 新的颜色。

    .. py:method:: set_poi(poi_x: float, poi_y: float)

        设置量子比特的位置坐标。

        参数：
            - **poi_x** (float) - 量子比特在绘制平面上的 x 坐标。
            - **poi_y** (float) - 量子比特在绘制平面上的 y 坐标。
