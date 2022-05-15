mindquantum.io.HiQASM.to_string(circuit, version='0.1')

        将电路转换为hiqasm。

        参数:
            circuit (Circuit): 要转换为HiQASM的量子电路。
            version (str): 要使用的HiQASM版本。默认值：“2.0”。

        返回:
            str，输入电路的HiQASM格式。

        异常:
            TypeError: 如果电路不是电路。
            TypeError: 如果版本不是str。
            NotImplementedError: 如果HiQASM版本未实现。
            ValueError: 如果在此版本中没有实现门。
        