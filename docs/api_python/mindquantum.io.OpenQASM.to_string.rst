.. py:method:: mindquantum.io.OpenQASM.to_string(circuit, version='2.0')

    将电路转换为openqasm。

    **参数：**

    - **circuit** (Circuit) - 要转换为openqasm的量子电路。
    - **version** (str) - 要使用的openqasm版本。默认值：“2.0”。

    **返回：**

    str，输入电路的openqasm格式。

    **异常：**

    - **TypeError** - 如果电路不是 `Circuit` 。
    - **TypeError** - 如果版本不是 `str` 。
    - **NotImplementedError** - 如果openqasm版本未实现。
    - **ValueError** - 如果在此版本中没有实现门。
