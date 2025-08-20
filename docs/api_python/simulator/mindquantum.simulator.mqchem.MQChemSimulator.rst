mindquantum.simulator.mqchem.MQChemSimulator
============================================

.. py:class:: mindquantum.simulator.mqchem.MQChemSimulator(n_qubits, n_electrons, seed=None, dtype="double")

    基于组态相互作用（CI）方法的量子化学模拟器。

    该模拟器通过在CI向量空间（由固定电子数定义的完整希尔伯特空间的子空间）中工作，针对化学问题进行了优化。
    对于典型的化学模拟，这种方法比使用完整的态矢量模拟器具有显著的性能优势。

    该模拟器设计用于与 :class:`~.simulator.mqchem.UCCExcitationGate` 和
    :class:`~.simulator.mqchem.CIHamiltonian` 协同工作。它提供了应用UCC线路、
    计算哈密顿量期望值以及计算变分量子算法（如VQE）所需梯度的方法。

    默认情况下，模拟器在Hartree-Fock态下初始化，该状态是量子化学计算的典型参考态。

    参数：
        - **n_qubits** (int) - 系统中的总量子比特数（自旋轨道数）。
        - **n_electrons** (int) - 电子数，定义了CI空间的维度。
        - **seed** (int) - 此模拟器的随机种子。如果为 ``None``，将生成一个随机种子。默认值：``None``。
        - **dtype** (str) - 模拟的数据类型，可以是 ``"float"`` 或 ``"double"``。默认值：``"double"``。

    .. py:method:: apply_circuit(circuit: Union[Circuit, Iterable[UCCExcitationGate]], pr: ParameterResolver = None)

        将量子线路应用于当前模拟器状态。

        .. note::
            线路中只有 :class:`~.simulator.mqchem.UCCExcitationGate` 的实例会被应用；所有其他类型的门都将被忽略。

        参数：
            - **circuit** (Union[Circuit, Iterable[UCCExcitationGate]]) - 要应用的量子线路或UCC门的可迭代对象。
            - **pr** (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]) - 用于替换参数值的参数解析器。如果为 ``None``，则直接使用门中定义的参数。默认值：``None``。

    .. py:method:: apply_gate(gate: UCCExcitationGate, pr: ParameterResolver = None)

        将单个UCC激发门应用于当前模拟器状态。

        参数：
            - **gate** (UCCExcitationGate) - 要应用的UCC激发门。
            - **pr** (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]) - 用于替换参数值的参数解析器。如果为 ``None``，则直接使用门中定义的参数。默认值：``None``。

        异常：
            - **TypeError** - 如果 `gate` 不是 :class:`~.simulator.mqchem.UCCExcitationGate`。

    .. py:method:: get_expectation(ham)

        计算哈密顿量相对于当前状态的期望值。

        计算 :math:`\langle\psi|H|\psi\rangle`，其中 :math:`|\psi\rangle` 是当前的CI态矢量，:math:`H` 是CI哈密顿量。

        参数：
            - **ham** (CIHamiltonian) - 要计算期望值的哈密顿量。

        返回：
            float，实数期望值。

        异常：
            - **TypeError** - 如果 `ham` 不是 :class:`~.simulator.mqchem.CIHamiltonian`。

    .. py:method:: get_expectation_with_grad(ham: CIHamiltonian, circuit: Union[Circuit, Iterable[UCCExcitationGate]])

        生成一个计算期望值及其梯度的函数。

        该方法实现了伴随微分法，以计算期望值 :math:`\langle\psi(\theta)|H|\psi(\theta)\rangle` 相对于UCC ansatz线路参数 :math:`\theta` 的梯度。
        状态被制备为 :math:`|\psi(\theta)\rangle = U(\theta)|\psi_0\rangle`，其中 :math:`|\psi_0\rangle` 是模拟器的当前状态。

        参数：
            - **ham** (CIHamiltonian) - 哈密顿量 :math:`H`。
            - **circuit** (Union[Circuit, Iterable[UCCExcitationGate]]) - 参数化的UCC线路 :math:`U(\theta)`。该线路必须具有用于梯度计算的参数。

        返回：
            Callable，一个接受参数值NumPy数组 `x` 并返回元组 `(expectation, gradient)` 的函数。`expectation` 是浮点数期望值，`gradient` 是一个NumPy数组，包含相对于 `x` 中每个参数的导数。参数的顺序由 `circuit.params_name` 决定。

        异常：
            - **TypeError** - 如果 `ham` 不是 :class:`~.simulator.mqchem.CIHamiltonian`。

    .. py:method:: get_qs(ket: bool = False)

        获取模拟器当前的量子态。

        虽然模拟器内部将状态存储为紧凑的CI向量，但此方法以完整的 :math:`2^{n_{qubits}}` 维计算基中的密集态矢量形式返回状态。

        参数：
            - **ket** (bool) - 如果为 ``True``，则以狄拉克（ket）符号的字符串形式返回量子态。如果为 ``False``，则以NumPy数组形式返回状态。默认值：``False``。

        返回：
            Union[numpy.ndarray, str]，量子态矢量，以NumPy数组或ket符号字符串形式表示。

        异常：
            - **TypeError** - 如果 `ket` 不是布尔值。

    .. py:method:: reset()

        将模拟器的状态重置为Hartree-Fock（HF）态。

        Hartree-Fock态是无相互作用费米子系统的基态，其中 `n_electrons` 个最低能量的自旋轨道被占据。在计算基中，这对应于状态 :math:`|11...100...0\rangle`。

    .. py:method:: set_qs(qs_rep)

        从稀疏表示设置CI向量。

        参数：
            - **qs_rep** (List[Tuple[int, complex]]) - 一个元组列表，其中每个元组 `(mask, amplitude)` 定义了基态的振幅。`mask` 是表示计算基态（斯莱特行列式）的整数，`amplitude` 是其对应的复振幅。`qs_rep` 中的所有基态的布居数必须等于 `n_electrons`。
