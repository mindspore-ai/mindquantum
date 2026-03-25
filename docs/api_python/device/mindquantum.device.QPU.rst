mindquantum.device.QPU
=======================

.. py:class:: mindquantum.device.QPU(n_qubits, default_shots=1000)

    量子处理单元（QPU）的执行基类。

    用户应实例化由硬件供应商提供的具体QPU子类（例如 ``MyVendorQPU(n_qubits, ...)``）。
    子类只需实现 :meth:`sampling` 方法；基类通过基于采样的方式提供了期望值和梯度的默认实现。

    .. note::
        - 默认的期望值/梯度实现仅支持由 :class:`~.core.operators.QubitOperator` 构造的
          :class:`~.core.operators.Hamiltonian` （硬件可测量的泡利和）。
        - 不支持非厄米期望值以及 ``circ_left`` / ``simulator_left`` 重叠形式。
        - 传入 :meth:`get_expectation` 或 :meth:`get_expectation_with_grad` 的线路不能包含
          测量门；测量基由内部自动生成。
        - 硬件约束（拓扑结构、原生门集、校准感知编译等）留给具体子类实现。

    参数：
        - **n_qubits** (int) - 该QPU上可用的量子比特数。
        - **default_shots** (int) - 当未显式提供 ``shots`` 时使用的默认测量次数。默认值： ``1000``。

    异常：
        - **TypeError** - 如果 `n_qubits` 或 `default_shots` 不是int。
        - **ValueError** - 如果 `n_qubits` 或 `default_shots` 小于1。

    .. py:method:: get_expectation(hamiltonian, circ_right=None, circ_left=None, simulator_left=None, pr=None, *, shots=None, seed=None)

        通过采样来估计哈密顿量的期望值。

        该方法与 :meth:`mindquantum.simulator.Simulator.get_expectation` 具有相同的用户接口，
        但期望值是通过测量采样来估计的，而非精确计算。

        仅支持厄米泡利和的情况：

        .. math::

            E = \langle 0 | U^\dagger H U | 0 \rangle

        其中 :math:`U` 是 ``circ_right``，:math:`|0\rangle` 表示隐式的全零初始态 :math:`|0\cdots0\rangle`。

        参数：
            - **hamiltonian** (Hamiltonian) - 需要计算期望的哈密顿量。必须由 :class:`~.core.operators.QubitOperator` 构造。
            - **circ_right** (Circuit) - 制备量子态的线路 :math:`U`。默认值： ``None`` （空线路）。
            - **circ_left** - 不支持。必须为 ``None``。
            - **simulator_left** - 不支持。必须为 ``None``。
            - **pr** (Union[Dict[str, numbers.Number], ParameterResolver, numpy.ndarray, list, numbers.Number]) - ``circ_right`` 的参数值。默认值： ``None``。

        关键字参数：
            - **shots** (int) - 每个泡利项测量线路的采样次数。如果为 ``None``，则使用 ``default_shots``。默认值： ``None``。
            - **seed** (int) - 可选的随机种子，传递给 :meth:`sampling`。默认值： ``None``。

        返回：
            numbers.Number，估计的期望值。返回虚部为零的复数，以与 :class:`~mindquantum.simulator.Simulator` 接口保持一致。

        异常：
            - **TypeError** - 如果 `hamiltonian` 不是 :class:`~.core.operators.Hamiltonian`。
            - **NotImplementedError** - 如果 `circ_left` 或 `simulator_left` 不为 ``None``，或者哈密顿量不是由 :class:`~.core.operators.QubitOperator` 构造的。
            - **ValueError** - 如果 `circ_right` 包含测量门，或者线路所需的量子比特数超过QPU提供的量子比特数。

    .. py:method:: get_expectation_with_grad(hams, circ_right, circ_left=None, simulator_left=None, parallel_worker=None, pr_shift=False, *, shots=None, seed=None, fd_gap=0.001)

        获取一个返回期望值和关于线路参数梯度的函数。

        返回的对象是 :class:`~mindquantum.simulator.GradOpsWrapper`，
        以保持与MindQuantum现有VQA / MindSpore集成的兼容性。

        对于真实硬件，梯度通过反复执行线路来估计：

        - 当 ``pr_shift=True`` 时，对于仅出现在一个梯度启用的内在参数位置、
          且其门匹配移位规则表（RX、RY、RZ、Rxx、Ryy、Rzz、Rxy、Rxz、Ryz、RPS、GP、PS、U3，
          以及具有特殊移位常数的SWAPalpha）的参数，使用参数移位规则。
          受控门和不支持的门（如Givens、FSim、自定义门）回退到有限差分法。
        - 当 ``pr_shift=False`` （默认）时，所有梯度均使用中心有限差分法。

        参数：
            - **hams** (Union[:class:`~.core.operators.Hamiltonian`, List[:class:`~.core.operators.Hamiltonian`]]) - 需要计算期望的 :class:`~.core.operators.Hamiltonian` 或一组 :class:`~.core.operators.Hamiltonian`。
            - **circ_right** (Circuit) - 变分线路 :math:`U`。
            - **circ_left** - 不支持。必须为 ``None``。
            - **simulator_left** - 不支持。必须为 ``None``。
            - **parallel_worker** (int) - 为保持接口兼容性而接受，但不使用。默认值： ``None``。
            - **pr_shift** (bool) - 是否在适用时优先使用参数移位规则。默认值： ``False``。

        关键字参数：
            - **shots** (int) - 每次线路执行的采样次数。如果为 ``None``，则使用 ``default_shots``。默认值： ``None``。
            - **seed** (int) - 可选的基础随机种子，用于梯度估计的可复现性。默认值： ``None``。
            - **fd_gap** (float) - 中心有限差分法的步长。默认值： ``0.001``。

        返回：
            GradOpsWrapper，一个可调用对象，其签名为 ``grad_ops(*inputs) -> (f, g_enc[, g_ans])``，其中 *f* 的形状为 ``(batch, n_hams)``，每个梯度数组的形状为 ``(batch, n_hams, n_params)``。

        异常：
            - **TypeError** - 如果 `hams` 不是Hamiltonian或Hamiltonian列表，或者 `circ_right` 不是Circuit。
            - **NotImplementedError** - 如果 `circ_left` 或 `simulator_left` 不为 ``None``，或者任何哈密顿量不是由 :class:`~.core.operators.QubitOperator` 构造的。
            - **ValueError** - 如果 `circ_right` 包含测量门，如果线路所需的量子比特数超过QPU提供的量子比特数，如果 `fd_gap` 不为正数，或者如果编码器参数和拟设参数集合存在重叠。

    .. py:method:: n_qubits
        :property:

        获取该QPU上可用的量子比特数。

        返回：
            int，量子比特数。

    .. py:method:: sampling(circuit, pr=None, shots=1, seed=None)
        :abstractmethod:

        在该QPU上执行 `circuit` 并返回采样结果。

        这是硬件后端 **必须** 实现的唯一方法。所有其他公共方法（期望值、梯度）均基于此方法构建。

        参数：
            - **circuit** (Circuit) - 包含至少一个测量门的量子线路。
            - **pr** (Union[dict, ParameterResolver]) - 含参线路的参数值。默认值： ``None``。
            - **shots** (int) - 测量次数。默认值： ``1``。
            - **seed** (int) - 可选的随机种子，用于结果可复现（真实硬件可能忽略此参数）。默认值： ``None``。

        返回：
            MeasureResult，采样的统计结果。
