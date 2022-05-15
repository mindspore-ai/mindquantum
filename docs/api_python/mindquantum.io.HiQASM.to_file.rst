mindquantum.io.HiQASM.to_file(file_name, circuit, version='0.1')

        将量子电路转换为HiQASM格式并保存在文件中。

        参数:
            file_name (str): 要保存HiQASM文件的文件名。
            circuit (Circuit): 要转换的电路。
            version (str): HiQASM的版本。默认值：“0.1”。

        异常:
            TypeError: 如果`file_name`不是str。
            TypeError: 如果`电路`不是电路。
            TypeError: 如果`version`不是str。
        