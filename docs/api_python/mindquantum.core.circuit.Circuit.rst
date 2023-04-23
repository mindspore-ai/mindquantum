mindquantum.core.circuit.Circuit
=================================

.. py:class:: mindquantum.core.circuit.Circuit(gates=None)

    量子线路模块。
    量子线路包含一个或多个量子门，可以在量子模拟器中进行计算。可以通过添加量子门或另一电路的方式容易地构建量子线路。

    参数：
        - **gates** (BasicGate, list[BasicGate]) - 可以通过单个量子门或门列表初始化量子线路。默认值： ``None``。

    .. py:method:: ansatz_params_name
        :property:

        获取线路中ansatz部分的参数名称。

        返回：
            list，线路中ansatz部分参数名称的list。

    .. py:method:: append(gate)

        增加一个门。

        参数：
            - **gate** (BasicGate) - 增加的门。

    .. py:method:: apply_value(pr)

        用输入的参数将该参数化量子线路转化为非参数量子线路。

        参数：
            - **pr** (Union[dict, ParameterResolver]) - 应用到此线路中的参数。

        返回：
            Circuit，不含参线路。

    .. py:method:: as_ansatz(inplace=True)

        将该量子线路变为ansatz量子线路。

        参数：
            - **inplace** (bool) - 是否原位设置。默认值： ``True``。

    .. py:method:: as_encoder(inplace=True)

        将该量子线路变为编码量子线路。

        参数：
            - **inplace** (bool) - 是否原位设置。默认值： ``True``。

    .. py:method:: barrier(show=True)

        添加barrier。

        参数：
            - **show** (bool) - 是否显示barrier。默认值： ``True``。

    .. py:method:: compress()

        删除所有未使用的量子比特，并将量子比特映射到 `range(n_qubits)` 。

    .. py:method:: encoder_params_name
        :property:

        获取线路中encoder部分的参数名称。

        返回：
            list，线路中encoder部分参数名称的list。

    .. py:method:: extend(gates)

        扩展线路。

        参数：
            - **gates** (Union[Circuit, list[BasicGate]]) - `Circuit` 或 `BasicGate` 的list。

    .. py:method:: get_cpp_obj(hermitian=False)

        获取线路的cpp object。

        参数：
            - **hermitian** (bool) - 是否获取线路cpp object的hermitian版本。默认值： ``False`` 。

    .. py:method:: get_qs(backend='mqvector', pr=None, ket=False, seed=None, dtype=None)

        获取线路的最终量子态。

        参数：
            - **backend** (str) - 使用的后端。默认值： ``'mqvector'``。
            - **pr** (Union[numbers.Number, ParameterResolver, dict, numpy.ndarray]) - 线路的参数，线路含参数时提供。默认值： ``None``。
            - **ket** (str) - 是否以ket格式返回量子态。默认值： ``False``。
            - **seed** (int) - 模拟器的随机种子。默认值： ``None``。
            - **dtype** (mindquantum.dtype) - 模拟器的数据类型。默认值： ``None``。

    .. py:method:: h(obj_qubits, ctrl_qubits=None)

        添加一个hadamard门。

        参数：
            - **obj_qubits** (Union[int, list[int]]) - `H` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `H` 门的控制量子比特。默认值： ``None`` 。

    .. py:method:: has_measure_gate
        :property:

        检查线路是否有测量门。

        返回：
            bool，线路是否有测量门。

    .. py:method:: hermitian()

        获得量子线路的厄米共轭。

    .. py:method:: insert(index, gates)

        在索引处插入量子门或量子线路。

        参数：
            - **index** (int) - 用来设置门的索引。
            - **gates** (Union[BasicGate, list[BasicGate]]) - 需要插入的量子门。

    .. py:method:: is_measure_end
        :property:

        检查线路是否以测量门结束，每个量子比特上最多有一个测量门，并且该测量门应位于该量子比特门序列的末尾。

        返回：
            bool，线路是否以测量门结束。

    .. py:method:: is_noise_circuit
        :property:

        检查线路是否有噪声信道。

        返回：
            bool，线路是否有噪声信道。

    .. py:method:: matrix(pr=None, big_end=False, backend='mqvector', seed=None, dtype=None)

        获取线路的矩阵表示。

        参数：
            - **pr** (ParameterResolver, dict, numpy.ndarray, list, numbers.Number) - 含参量子线路的参数。默认值： ``None``。
            - **big_end** (bool) - 低索引量子比特是否放置在末尾。默认值： ``False``。
            - **backend** (str) - 进行模拟的后端。默认值： ``'mqvector'``。
            - **seed** (int) - 生成线路矩阵的随机数，如果线路包含噪声信道。
            - **dtype** (mindquantum.dtype) - 模拟器的数据类型。默认值： ``None``。

        返回：
            numpy.ndarray，线路的二维复矩阵。

    .. py:method:: measure(key, obj_qubit=None)

        添加一个测量门。

        参数：
            - **key** (Union[int, str]) - 如果 `obj_qubit` 为 ``None`` ，则 `key` 应为int，表示要测量哪个量子比特，否则， `key` 应为str，表示测量门的名称。
            - **obj_qubit** (int) - 要测量的量子比特。默认值： ``None``。

    .. py:method:: measure_all(suffix=None)

        测量所有量子比特。

        参数：
            - **suffix** (str) - 添加到测量门名称中的后缀字符串。

    .. py:method:: n_qubits
        :property:

        获取量子线路所使用的比特数。

    .. py:method:: no_grad()

        设置量子线路中所有不需要梯度的含参门。

    .. py:method:: parameter_resolver()

        获取整个线路的parameter resolver。

        .. note::
            因为相同的参数可以在不同的门中，并且系数可以不同，所以这个parameter resolver只返回量子线路的参数是什么，哪些参数需要梯度。显示系数的更详细的parameter resolver位于线路的每个门中。

        返回：
            ParameterResolver，整个线路的parameter resolver。

    .. py:method:: parameterized
        :property:

        检查线路是否是含参量子线路。

        返回：
            bool，线路是否是含参量子线路。

    .. py:method:: params_name
        :property:

        获取线路的参数名称。

        返回：
            list，包含参数名称的list。

    .. py:method:: phase_shift(para, obj_qubits, ctrl_qubits=None)

        添加一个PhaseShift门。

        参数：
            - **para** (Union[dict, ParameterResolver]) - `PhaseShift` 门的参数。
            - **obj_qubits** (Union[int, list[int]]) - `PhaseShift` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `PhaseShift` 门的控制量子比特。默认值： ``None``。

    .. py:method:: remove_barrier()

        移除所有barrier门。

    .. py:method:: remove_measure()

        移除所有的测量门。

    .. py:method:: remove_measure_on_qubits(qubits)

        移除某些量子比特上所有的测量门。

        参数：
            - **qubit** (Union[int, list[int]]) - 需要删除测量门的量子比特。

    .. py:method:: remove_noise()

        删除量子线路中的所有噪声信道。

    .. py:method:: requires_grad()

        将量子线路中的所有含参门都设置为需要梯度。

    .. py:method:: reverse_qubits()

        将线路翻转成大端头(big endian)。

    .. py:method:: rx(para, obj_qubits, ctrl_qubits=None)

        在电路中添加 `RX` 门。

        参数：
            - **para** (Union[dict, ParameterResolver]) - `RX` 门的参数。
            - **obj_qubits** (Union[int, list[int]]) - `RX` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `RX` 门的控制量子比特。默认值： ``None`` 。

    .. py:method:: rxx(para, obj_qubits, ctrl_qubits=None)

        在电路中添加 `Rxx` 门。

        参数：
            - **para** (Union[dict, ParameterResolver]) - `Rxx` 门的参数。
            - **obj_qubits** (Union[int, list[int]]) - `Rxx` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `Rxx` 门的控制量子比特。默认值： ``None``。

    .. py:method:: ry(para, obj_qubits, ctrl_qubits=None)

        在电路中添加 `RY` 门。

        参数：
            - **para** (Union[dict, ParameterResolver]) - `RY` 门的参数。
            - **obj_qubits** (Union[int, list[int]]) - `RY` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `RY` 门的控制量子比特。默认值： ``None`` 。

    .. py:method:: ryy(para, obj_qubits, ctrl_qubits=None)

        在电路中添加 `Ryy` 门。

        参数：
            - **para** (Union[dict, ParameterResolver]) - `Ryy` 门的参数。
            - **obj_qubits** (Union[int, list[int]]) - `Ryy` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `Ryy` 门的控制量子比特。默认值： ``None``。

    .. py:method:: rz(para, obj_qubits, ctrl_qubits=None)

        在电路中添加 `RZ` 门。

        参数：
            - **para** (Union[dict, ParameterResolver]) - `RZ` 门的参数。
            - **obj_qubits** (Union[int, list[int]]) - `RZ` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `RZ` 门的控制量子比特。默认值： ``None``。

    .. py:method:: rzz(para, obj_qubits, ctrl_qubits=None)

        在电路中添加 `Rzz` 门。

        参数：
            - **para** (Union[dict, ParameterResolver]) - `Rzz` 门的参数。
            - **obj_qubits** (Union[int, list[int]]) - `Rzz` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `Rzz` 门的控制量子比特。默认值： ``None``。

    .. py:method:: s(obj_qubits, ctrl_qubits=None)

        在电路中添加 `S` 门。

        参数：
            - **obj_qubits** (Union[int, list[int]]) - `S` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `S` 门的控制量子比特。默认值： ``None``。

    .. py:method:: summary(show=True)

        打印当前线路的信息，包括块的数量、门的数量、不含参门的数量、含参门的数量和参数的个数。

        参数：
            - **show** (bool) - 是否显示信息。默认值： ``True``。

    .. py:method:: svg(style=None, width=None)

        在Jupyter Notebook中将当前量子线路用SVG图展示。

        参数：
            - **style** (dict, str) - 设置svg线路的样式。目前，我们支持'official'，'light'和'dark'。默认值： ``None``。
            - **width** (int, float) - 设置量子线路的最大宽度。默认值： ``None``。

    .. py:method:: swap(obj_qubits, ctrl_qubits=None)

        在电路中添加 `SWAP` 门。

        参数：
            - **obj_qubits** (Union[int, list[int]]) - `SWAP` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `SWAP` 门的控制量子比特。默认值： ``None``。

    .. py:method:: un(gate, maps_obj, maps_ctrl=None)

        将量子门作用于多个目标量子比特和控制量子比特，详见类 :class:`mindquantum.core.circuit.UN` 。

        参数：
            - **gate** (BasicGate) - 要执行的量子门。
            - **maps_obj** (Union[int, list[int]]) - 执行该量子门的目标量子比特。
            - **maps_ctrl** (Union[int, list[int]]) - 执行该量子门的控制量子比特。默认值： ``None``。

    .. py:method:: with_noise(noise_gate=mq_gates.AmplitudeDampingChannel(0.001), also_ctrl=False)

        在每个量子门后面添加一个噪声信道。

        参数：
            - **noise_gate** (NoiseGate) - 想要添加的噪声信道。默认值：``AmplitudeDampingChannel(0.001)``。
            - **also_ctrl** (bool) - 是否在控制比特上也加噪声信道。默认值：``False``。

    .. py:method:: x(obj_qubits, ctrl_qubits=None)

        在电路中添加 `X` 门。

        参数：
            - **obj_qubits** (Union[int, list[int]]) - `X` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `X` 门的控制量子比特。默认值： ``None``。

    .. py:method:: xx(para, obj_qubits, ctrl_qubits=None)

        在电路中添加 `XX` 门。

        .. note::
            `xx` 方法已弃用，请使用 :class:`mindquantum.core.circuit.Circuit.rxx`。

        参数：
            - **para** (Union[dict, ParameterResolver]) - `XX` 门的参数。
            - **obj_qubits** (Union[int, list[int]]) - `XX` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `XX` 门的控制量子比特。默认值： ``None``。

    .. py:method:: y(obj_qubits, ctrl_qubits=None)

        在电路中添加 `Y` 门。

        参数：
            - **obj_qubits** (Union[int, list[int]]) - `Y` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `Y` 门的控制量子比特。默认值： ``None``。

    .. py:method:: yy(para, obj_qubits, ctrl_qubits=None)

        在电路中添加 `YY` 门。

        .. note::
            `yy` 方法已弃用，请使用 :class:`mindquantum.core.circuit.Circuit.ryy`。

        参数：
            - **para** (Union[dict, ParameterResolver]) - `YY` 门的参数。
            - **obj_qubits** (Union[int, list[int]]) - `YY` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `YY` 门的控制量子比特。默认值： ``None``。

    .. py:method:: z(obj_qubits, ctrl_qubits=None)

        在电路中添加 `Z` 门。

        参数：
            - **obj_qubits** (Union[int, list[int]]) - `Z` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `Z` 门的控制量子比特。默认值： ``None``。

    .. py:method:: zz(para, obj_qubits, ctrl_qubits=None)

        在电路中添加 `ZZ` 门。

        .. note::
            `zz` 方法已弃用，请使用 :class:`mindquantum.core.circuit.Circuit.rzz`。

        参数：
            - **para** (Union[dict, ParameterResolver]) - `ZZ` 门的参数。
            - **obj_qubits** (Union[int, list[int]]) - `ZZ` 门的目标量子比特。
            - **ctrl_qubits** (Union[int, list[int]]) - `ZZ` 门的控制量子比特。默认值： ``None``。
