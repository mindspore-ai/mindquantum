.. py:method:: mindquantum.io.HiQASM.to_file(file_name, circuit, version='0.1')

    将量子线路转换为HiQASM格式并保存在文件中。

    **参数：**

    - **file_name** (str) - 需要保存成HiQASM文件的文件名。
    - **circuit** (Circuit) - 需要转换的线路。
    - **version** (str) - HiQASM的版本。默认值："0.1"。

    **异常：**

    - **TypeError** - 如果 `file_name` 类型不是str。
    - **TypeError** - 如果 `Circuit` 类型不是circuit。
    - **TypeError** - 如果 `version` 类型不是str。
