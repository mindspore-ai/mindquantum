mindquantum.simulator.Simulator
================================

.. py:class:: mindquantum.simulator.Simulator(backend, n_qubits, seed=None, dtype=None, *args, **kwargs)

    模拟量子线路的量子模拟器。

    参数：
        - **backend** (str) - 想要的后端。通过调用 `get_supported_simulator()` 可以返回支持的后端。
        - **n_qubits** (int) - 量子模拟器的量子比特数量。
        - **seed** (int) - 模拟器的随机种子，如果为 ``None``，种子将由 `numpy.random.randint` 生成。默认值： ``None``。
        - **dtype** (mindquantum.dtype) - 模拟器的数据类型。

    异常：
        - **TypeError** - 如果 `backend` 不是str。
        - **TypeError** - 如果 `n_qubits` 不是int。
        - **TypeError** - 如果 `seed` 不是int。
        - **ValueError** - 如果不支持 `backend` 。
        - **ValueError** - 如果 `n_qubits` 为负数。
        - **ValueError** - 如果 `seed` 小于0或大于 :math:`2^23 - 1` 。

    .. py:method:: astype(dtype, seed=None)

        将模拟器转化给定的数据类型。

        参数：
            - **dtype** (mindquantum.dtype) - 新模拟器的数据类型。
            - **seed** (int) - 新模拟器的随机数种子。默认值： ``None``。

    .. py:method:: apply_circuit(circuit, pr=None)

        在模拟器上应用量子线路。

        参数：
            - **circuit** (Circuit) - 要应用在模拟器上的量子线路。
            - **pr** (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]) - 线路的ParameterResolver。如果线路不含参数，则此参数应为None。默认值： ``None``。

        返回：
            MeasureResult或None，如果线路具有测量门，则返回MeasureResult，否则返回None。

    .. py:method:: apply_gate(gate, pr=None, diff=False)

        在此模拟器上应用门，可以是量子门或测量算子。

        参数：
            - **gate** (BasicGate) - 要应用的门。
            - **pr** (Union[numbers.Number, numpy.ndarray, ParameterResolver, list]) - 含参门的参数。默认值： ``None``。
            - **diff** (bool) - 是否在模拟器上应用导数门。默认值： ``False``。

        返回：
            int或None，如果是该门是测量门，则返回坍缩态，否则返回None。

        异常：
            - **TypeError** - 如果 `gate` 不是BasicGate。
            - **ValueError** - 如果 `gate` 的某个量子比特大于模拟器本身的量子比特。
            - **ValueError** - 如果 `gate` 是含参的，但没有提供参数。
            - **TypeError** - 如果 `gate` 是含参的，但 `pr` 不是ParameterResolver。

    .. py:method:: apply_hamiltonian(hamiltonian: Hamiltonian)

        将hamiltonian应用到模拟器上，这个hamiltonian可以是hermitian或non hermitian。

        .. note::
            应用hamiltonian后，量子态可能不是归一化量子态。

        参数：
            - **hamiltonian** (Hamiltonian) - 想应用的hamiltonian。

    .. py:method:: copy()

        复制模拟器。

        返回：
            模拟器，当前模拟器的副本。

    .. py:method:: dtype
        :property:

        返回模拟器的数据类型。

    .. py:method:: get_expectation(hamiltonian, circ_right=None, circ_left=None, simulator_left=None, pr=None)

        得到给定hamiltonian的期望。hamiltonian可能是非厄米共轭的。该方法旨在计算期望值，如下所示：

        .. math::

            E = \left<\varphi\right|U_l^\dagger H U_r \left|\psi\right>

        参数：
            - **hamiltonian** (Hamiltonian) - 想得到期望的hamiltonian。
            - **circ_right** (Circuit) - 表示 :math:`U_r` 的线路。如果为 ``None``，则选择空线路。默认值： ``None``。
            - **circ_left** (Circuit) - 表示 :math:`U_l` 的线路。如果为 ``None``，则将设置成 ``circ_right`` 一样的线路。默认值： ``None``。
            - **simulator_left** (Simulator) - 包含 :math:`\left|\varphi\right>` 的模拟器。如果无，则 :math:`\left|\varphi\right>` 被假定等于 :math:`\left|\psi\right>`。默认值： ``None``。
            - **pr** (Union[Dict[str, numbers.Number], ParameterResolver]) - 线路中的参数。默认值： ``None``.

        返回：
            numbers.Number，期望值。

    .. py:method:: get_expectation_with_grad(hams, circ_right, circ_left=None, simulator_left=None, parallel_worker=None)

        获取一个返回前向值和关于线路参数梯度的函数。该方法旨在计算期望值及其梯度，如下所示：

        .. math::

            E = \left<\varphi\right|U_l^\dagger H U_r \left|\psi\right>

        其中 :math:`U_l` 是circ_left，:math:`U_r` 是circ_right，:math:`H` 是hams，:math:`\left|\psi\right>` 是模拟器当前的量子态，:math:`\left|\varphi\right>` 是 `simulator_left` 的量子态。

        参数：
            - **hams** (Hamiltonian) - 需要计算期望的Hamiltonian。
            - **circ_right** (Circuit) - 上述 :math:`U_r` 电路。
            - **circ_left** (Circuit) - 上述 :math:`U_l` 电路，默认情况下，这个线路将为 ``None``，在这种情况下， :math:`U_l` 将等于 :math:`U_r` 。默认值： ``None``。
            - **simulator_left** (Simulator) - 包含 :math:`\left|\varphi\right>` 的模拟器。如果无，则 :math:`\left|\varphi\right>` 被假定等于 :math:`\left|\psi\right>`。默认值： ``None``。
            - **parallel_worker** (int) - 并行器数目。并行器可以在并行线程中处理batch。默认值： ``None``。

        返回：
            GradOpsWrapper，一个包含生成梯度算子信息的梯度算子包装器。

    .. py:method:: get_qs(ket=False)

        获取模拟器的当前量子态。

        参数：
            - **ket** (bool) - 是否以ket格式返回量子态。默认值： ``False``。

        返回：
            numpy.ndarray，当前量子态。

    .. py:method:: n_qubits()

        获取模拟器的量子比特数。

        返回：
            int，当前模拟器的量子比特数。

    .. py:method:: reset()

        将模拟器重置为0态。

    .. py:method:: sampling(circuit, pr=None, shots=1, seed=None)

        在线路中对测量比特进行采样。采样不会改变模拟器的量子态。

        参数：
            - **circuit** (Circuit) - 要进行演化和采样的电路。
            - **pr** (Union[None, dict, ParameterResolver]) - 线路的parameter resolver，如果线路是含参线路则需要提供pr。默认值： ``None``。
            - **shots** (int) - 采样线路的次数。默认值： ``1``。
            - **seed** (int) - 采样的随机种子。如果为None，则种子将是随机的整数。默认值： ``None``。

        返回：
            MeasureResult，采样的统计结果。

    .. py:method:: set_qs(quantum_state)

        设置模拟器的量子态。

        参数：
            - **quantum_state** (numpy.ndarray) - 想设置的量子态。

    .. py:method:: set_threads_number(number)

        设置最大线程数。

        参数：
            - **number** (int) - 设置模拟器中线程池所使用的线程数。
