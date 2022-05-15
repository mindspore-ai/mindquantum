mindquantum.core.circuit.Circuit.get_qs(backend='projectq', pr=None, ket=False, seed=None)

        获取此电路的最终量子状态。

        参数:
            backend (str): 您要使用的后端。默认值：'项目q'。
            pr (Union[numbers.Number, ParameterResolver, dict, numpy.ndarray]): 此电路的参数，如果此电路被参数化。默认值：None。
            ket (str): 是否以ket格式返回量子状态。默认值：False。
            seed (int): 模拟器的随机种子。默认值：None。
        