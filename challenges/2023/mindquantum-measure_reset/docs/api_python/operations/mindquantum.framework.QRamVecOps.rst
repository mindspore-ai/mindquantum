mindquantum.framework.QRamVecOps
================================

.. py:class:: mindquantum.framework.QRamVecOps(hams, circ, sim, n_thread=None)

    QRam 算子，该算子可以直接将经典数据编码为全振幅量子态。此算子只能在 `PYNATIVE_MODE` 下执行。

    .. note::
        - 由于 MindSpore 小于 2.0.0 版本重点神经网络不支持复数作为输入，所以我们将量子态的实部和虚部分开，分别作为输入参数输入到量子神经网络中。当 MindSpore 升级时，这一行为有可能会改变。
        - 当前，我们不能计算测量结果关于量子态概率幅的导数。

    参数：
        - **hams** (Union[:class:`~.core.operators.Hamiltonian`, List[:class:`~.core.operators.Hamiltonian`]]) - 要想求期望值的哈密顿量或者一组哈密顿量。
        - **circ** (:class:`~.core.circuit.Circuit`) - 变分量子线路。
        - **sim** (:class:`~.simulator.Simulator`) - 做模拟所使用到的模拟器。
        - **n_thread** (int) - 运行一个batch的初始态时的并行数。如果是 ``None``，用单线程来运行。默认值： ``None``。

    输入：
        - **qs_r** (Tensor) - 量子态实部的Tensor，其shape为 :math:`(N, M)` ，其中 :math:`N` 表示batch大小， :math:`M` 表示全振幅量子态的长度。
        - **qs_i** (Tensor) - 量子态虚部的Tensor，其shape为 :math:`(N, M)` ，其中 :math:`N` 表示batch大小， :math:`M` 表示全振幅量子态的长度。
        - **ans_data** (Tensor) - shape为 :math:`N` 的Tensor，用于ansatz电路，其中 :math:`N` 表示ansatz参数的数量。

    输出：
        Tensor，hamiltonian的期望值。
