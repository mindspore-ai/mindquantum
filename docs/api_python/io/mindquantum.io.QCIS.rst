mindquantum.io.QCIS
========================

.. py:class:: mindquantum.io.QCIS

    将电路转换为qcis格式的模块。

    .. py:method:: from_file(file_name: str)

        读取qcis文件。

        参数：
            - **file_name** (str) - 以qcis格式存储量子线路的文件路径。

        返回：
            Circuit，从qcis文件翻译过来的量子线路。

    .. py:method:: from_string(string: str)

        读取 QCIS 字符串。

        参数：
            - **string** (str) - 量子线路的 QCIS 字符串表示。

        返回：
            :class:`~.core.circuit.Circuit`，QCIS 字符串表示的量子线路。

    .. py:method:: to_file(file_name: str, circuit, parametric: bool = True)

        将量子线路转换为qcis格式并保存在文件中。

        参数：
            - **file_name** (str) - 要保存qcis文件的文件名。
            - **circuit** (Circuit) - 要转换的电路。
            - **parametric** (bool) - 是否保留参数。如果为 ``False``，则会丢弃所有参数，以及参数值为 0 的参数门，参数门的角度也会被限制在区间 [-pi, pi]。默认值： ``True``。

        异常：
            - **TypeError** - 如果 `file_name` 不是 `str` 。
            - **TypeError** - 如果 `circuit` 不是 `Circuit` 。
            - **NotImplementedError** - 如果 `circuit` 中包含QCIS不支持的量子门。

    .. py:method:: to_string(circuit, parametric: bool = True)

        将电路转换为qcis。

        参数：
            - **circuit** (Circuit) - 要转换为qcis的量子线路。
            - **parametric** (bool) - 是否保留参数。如果为 ``False``，则会丢弃所有参数，以及参数值为 0 的参数门，参数门的角度也会被限制在区间 [-pi, pi]。默认值： ``True``。

        返回：
            str，输入电路的qcis格式。

        异常：
            - **TypeError** - 如果电路不是 `Circuit` 。
            - **NotImplementedError** - 如果 `circuit` 中包含QCIS不支持的量子门。
