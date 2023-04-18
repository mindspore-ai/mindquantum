mindquantum.io.HiQASM
======================

.. py:class:: mindquantum.io.HiQASM()

    将线路转换为HiQASM格式。

    .. py:method:: from_file(file_name)

        读取HiQASM文件。

        参数：
            - **file_name** (str) - 以HiQASM格式存储量子线路的文件路径。

        返回：
            线路，从HiQASM文件转换过来的量子线路。

    .. py:method:: from_string(string)

        读取HiQASM字符串。

        参数：
            - **string** (str) - 线路的HiQASM字符串。

        返回：
            线路，从HiQASM字符串转换过来的量子线路。

    .. py:method:: to_file(file_name, circuit, version='0.1')

        将量子线路转换为HiQASM格式并保存在文件中。

        参数：
            - **file_name** (str) - 需要保存成HiQASM文件的文件名。
            - **circuit** (Circuit) - 需要转换的线路。
            - **version** (str) - HiQASM的版本。默认值： ``"0.1"``。

        异常：
            - **TypeError** - 如果 `file_name` 类型不是str。
            - **TypeError** - 如果 `Circuit` 类型不是circuit。
            - **TypeError** - 如果 `version` 类型不是str。

    .. py:method:: to_string(circuit, version='0.1')

        将量子线路转换为HiQASM。

        参数：
            - **circuit** (Circuit) - 需要转换为HiQASM的量子线路。
            - **version** (str) - 需要使用的HiQASM版本。默认值： ``"0.1"``。

        返回：
            str，输入线路对应的HiQASM格式。

        异常：
            - **TypeError** - 如果 `Circuit` 类型不是circuit。
            - **TypeError** - 如果 `version` 类型不是str。
            - **NotImplementedError** - 如果HiQASM版本未实现。
            - **ValueError** - 如果在此版本中没有实现门。
