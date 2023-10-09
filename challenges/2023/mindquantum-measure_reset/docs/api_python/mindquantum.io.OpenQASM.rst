mindquantum.io.OpenQASM
========================

.. py:class:: mindquantum.io.OpenQASM

    将电路转换为openqasm格式的模块。

    .. py:method:: from_file(file_name)

        读取openqasm文件。

        参数：
            - **file_name** (str) - 以openqasm格式存储量子线路的文件路径。

        返回：
            Circuit，从openqasm文件翻译过来的量子线路。

    .. py:method:: from_string(string)

        读取 OpenQASM 字符串。

        参数：
            - **string** (str) - 量子线路的 OpenQASM 字符串表示。

        返回：
            :class:`~.core.circuit.Circuit`，OpenQASM 字符串表示的量子线路。

    .. py:method:: to_file(file_name, circuit, version='2.0')

        将量子线路转换为openqasm格式并保存在文件中。

        参数：
            - **file_name** (str) - 要保存openqasm文件的文件名。
            - **circuit** (Circuit) - 要转换的电路。
            - **version** (str) - openqasm的版本。默认值： ``"2.0"``。

        异常：
            - **TypeError** - 如果 `file_name` 不是 `str` 。
            - **TypeError** - 如果 `电路` 不是 `Circuit` 。
            - **TypeError** - 如果 `version` 不是 `str` 。

    .. py:method:: to_string(circuit, version='2.0')

        将电路转换为openqasm。

        参数：
            - **circuit** (Circuit) - 要转换为openqasm的量子线路。
            - **version** (str) - 要使用的openqasm版本。默认值： ``"2.0"``。

        返回：
            str，输入电路的openqasm格式。

        异常：
            - **TypeError** - 如果电路不是 `Circuit` 。
            - **TypeError** - 如果版本不是 `str` 。
            - **NotImplementedError** - 如果openqasm版本未实现。
            - **ValueError** - 如果在此版本中没有实现门。
