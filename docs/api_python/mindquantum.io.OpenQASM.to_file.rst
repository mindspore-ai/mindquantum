.. py:method:: mindquantum.io.OpenQASM.to_file(file_name, circuit, version='2.0')

    将量子电路转换为openqasm格式并保存在文件中。

    **参数：**

    - **file_name** (str) - 要保存openqasm文件的文件名。
    - **circuit** (Circuit) - 要转换的电路。
    - **version** (str) - openqasm的版本。默认值："2.0"。

    **异常：**

    - **TypeError** - 如果 `file_name` 不是 `str` 。
    - **TypeError** - 如果 `电路` 不是 `Circuit` 。
    - **TypeError** - 如果 `version` 不是 `str` 。
