mindquantum.core.circuit.Circuit.matrix(pr=None, big_end=False, backend='projectq', seed=None)

        获取此电路的矩阵。

        参数:
            pr (ParameterResolver, dict, numpy.ndarray, list, numbers.Number): 参数化量子电路的参数解析器。默认值：None。
            big_end (bool): 低索引量子位是否放置在末尾。默认值：False。
            backend (str): 要进行模拟的后端。默认值：'projectq'。
            seed (int): 如果电路具有噪声信道，则生成电路矩阵的随机数。

        样例:
            >>> from mindquantum.core import Circuit
            >>> circuit = Circuit().rx('a',0).h(0)
            >>> circuit.matrix({'a': 1.0})
            array([[ 0.62054458-0.33900505j,  0.62054458-0.33900505j],
                [ 0.62054458+0.33900505j, -0.62054458-0.33900505j]])

        返回:
            numpy.nd数组，本电路的二维复矩阵。
        