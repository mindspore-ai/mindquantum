mindquantum.core.gates.MeasureResult
======================================

.. py:class:: mindquantum.core.gates.MeasureResult

    测量结果容器。

    .. py:method:: add_measure(measure)

        在此测量结果容器中添加测量门。测量键在此测量结果容器中应是唯一的。

        参数：
            - **measure** (Union[Iterable, Measure]) - 一个或多个测量门。

    .. py:method:: collect_data(samples)

        收集所有测量门测量出的比特串。

        参数：
            - **samples** (numpy.ndarray) - 一个二维(N x M) numpy数组，以0或1存储采样位字符串，其中N表示拍摄次数，M表示此测量容器中的键数。

    .. py:method:: data
        :property:

        获取采样数据。

        返回：
            dict，采样数据。

    .. py:method:: keys_map
        :property:

        返回测量门名词与出现顺序的关系的字典。

    .. py:method:: select_keys(*keys)

        从该测量容器中选择某些测量键。

        参数：
            - **keys** (tuple[str]) - 要选择的键。

    .. py:method:: svg(style=None)

        将当前测量结果显示为jupyter notebook中的SVG图片。

        参数：
            - **style** (dict, str) - 设置svg样式的样式。目前，我们支持 ``'official'``。默认值： ``None``。
