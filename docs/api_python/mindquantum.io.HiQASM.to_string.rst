.. py:method:: mindquantum.io.HiQASM.to_string(circuit, version='0.1')

    将量子线路转换为HiQASM。

    **参数：**

    - **circuit** (Circuit) – 需要转换为HiQASM的量子线路。
    - **version** (str) – 需要使用的HiQASM版本。默认值："0.1"。

    **返回：**

    str，输入线路对应的HiQASM格式。

    **异常：**

    - **TypeError** – 如果 `Circuit` 类型不是circuit。
    - **TypeError** – 如果 `version` 类型不是str。
    - **NotImplementedError** – 如果HiQASM版本未实现。
    - **ValueError** – 如果在此版本中没有实现门。
