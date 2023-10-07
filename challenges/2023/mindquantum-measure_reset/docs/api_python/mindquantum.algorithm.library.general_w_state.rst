mindquantum.algorithm.library.general_w_state
==============================================

.. py:function:: mindquantum.algorithm.library.general_w_state(qubits)

    通用W态。
    W态通常定义成只有单个比特是 :math:`\left|1\right>` 态的基矢的均匀叠加，而其他态都为 :math:`\left|0\right>` 。举个例子，对于三量子比特系统，W态定义为：

    .. math::

        \left|\rm W\right> = (\left|001\right> + \left|010\right> + \left|100\right>)/\sqrt(3)

    在本接口中，我们可以定义任意总量子比特系统中任意部分希尔伯特空间中的W态。

    .. note::
        请参考 https://quantumcomputing.stackexchange.com/questions/4350/general-construction-of-w-n-state。

    参数：
        - **qubits** (list[int]) - 需要应用通用W态的量子比特。

    返回：
        Circuit，可以制备W态的线路。
